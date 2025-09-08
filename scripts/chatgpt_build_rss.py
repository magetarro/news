#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import math
import html
import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
import feedparser
import pytz
from dateutil import parser as dtparser
from lxml import etree
import yaml

# -----------------------------
# Config & Env
# -----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
OPENAI_RETRIES = int(os.getenv("OPENAI_RETRIES", "3"))
OPENAI_BACKOFF = float(os.getenv("OPENAI_BACKOFF", "2.0"))

USER_AGENT = "news-rss-bot/1.0 (+github-actions)"

# лимиты для подготовки данных к LLM
MAX_PER_MODE = int(os.getenv("MAX_PER_MODE", "5"))           # общее ограничение постов на режим
MAX_PER_CHANNEL = int(os.getenv("MAX_PER_CHANNEL", "4"))     # постов с одного канала
MAX_CHARS_PER_ITEM = int(os.getenv("MAX_CHARS_PER_ITEM", "700"))
LLM_PER_CHUNK = int(os.getenv("LLM_PER_CHUNK", "5"))

# временные настройки
DEFAULT_TZ = "America/Los_Angeles"

# исключение .ru и пр.
ALLOWED_TLDS_EXCLUDE = (".ru",)
BLACKLIST_HOSTS = {
    "ria.ru", "sputniknews.com", "rt.com", "tass.ru"
}

# ключевые слова для редактирования донатов
FUNDRAISING_KEYWORDS = [
    "донат", "донейт", "донатить", "донатів", "пожертв", "сбор средств",
    "збір коштів", "monobank", "mono банку", "privat24", "приват24",
    "карта", "реквизит", "реквізит", "qiwi", "patreon", "buymeacoffee",
    "募金", "donate", "fundraiser", "fundraising"
]


# -----------------------------
# Helpers
# -----------------------------

def http_get(url, timeout=30, headers=None):
    headers = headers or {}
    h = {"User-Agent": USER_AGENT}
    h.update(headers)
    resp = requests.get(url, timeout=timeout, headers=h)
    resp.raise_for_status()
    return resp


def fetch_rss(url, max_retries=4):
    """Загрузка RSS с простыми ретраями (обработка 429 и 5xx)."""
    backoff = 2.0
    last_exc = None
    for i in range(max_retries):
        try:
            r = http_get(url, timeout=30)
            feed = feedparser.parse(r.content)
            entries_count = len(getattr(feed, "entries", []) or [])
            print(f"[fetch_rss] OK {r.status_code} {url} entries={entries_count}")
            return feed
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                sleep_s = round(backoff + (i * 0.5), 1)
                print(f"[fetch_rss] 429, sleep {sleep_s}s {url}")
                time.sleep(sleep_s)
                backoff *= 2.0
                last_exc = e
                continue
            elif 500 <= status < 600:
                sleep_s = round(backoff, 1)
                print(f"[fetch_rss] {status}, sleep {sleep_s}s {url}")
                time.sleep(sleep_s)
                backoff *= 2.0
                last_exc = e
                continue
            else:
                print(f"[fetch_rss] FAIL {url}: {e}")
                last_exc = e
                break
        except Exception as e:
            print(f"[fetch_rss] FAIL {url}: {e}")
            last_exc = e
            time.sleep(1.0)
    if last_exc:
        raise last_exc
    return None


def parse_time(entry):
    """Определяем pubDate в UTC."""
    # feedparser уже парсит published_parsed/updated_parsed
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        return dt
    # как fallback — пытаемся из полей в виде строки
    for k in ("published", "updated", "created"):
        v = getattr(entry, k, None)
        if v:
            try:
                dt = dtparser.parse(v)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    # если вообще нет — берём сейчас
    return datetime.now(timezone.utc)


def now_local(tzname):
    tz = pytz.timezone(tzname)
    return datetime.now(tz)


def to_rfc822(dt):
    if isinstance(dt, str):
        dt = dtparser.parse(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime('%a, %d %b %Y %H:%M:%S %z')


def strip_html(s):
    if not s:
        return ""
    # убираем теги
    txt = re.sub(r"<[^>]+>", "", s)
    # unescape
    return html.unescape(txt).strip()


def is_russia_affiliated(link: str) -> bool:
    if not link:
        return False
    try:
        host = re.sub(r"^https?://", "", link).split("/")[0].lower()
    except Exception:
        return False
    if any(host.endswith(tld) for tld in ALLOWED_TLDS_EXCLUDE):
        return True
    for bad in BLACKLIST_HOSTS:
        if bad in host:
            return True
    return False


def redact_fundraising_text(txt: str) -> str:
    """Удаляем фразы/абзацы с упоминаниями донатов, сохраняя остальной текст."""
    if not txt:
        return txt
    # удаляем строки/абзацы с ключевиками
    lines = re.split(r"(\r?\n)+", txt)
    out = []
    for line in lines:
        low = line.lower()
        if any(k in low for k in FUNDRAISING_KEYWORDS):
            continue
        out.append(line)
    res = "".join(out)
    # блюрим явные номера карт и короткие платежные ссылки
    res = re.sub(r"\b\d{12,19}\b", "[REDACTED]", res)   # номера карт
    res = re.sub(r"(t\.me\/\+?donate[^\s]*)", "[REDACTED]", res, flags=re.I)
    return res.strip()


# -----------------------------
# Telegram gather (via RSSHub URLs from config)
# -----------------------------

def load_config():
    # пытаемся прочитать sources.yml из корня репо
    path = os.path.join(os.getcwd(), "sources.yml")
    if not os.path.exists(path):
        # дефолтная конфигурация (минимальная)
        return {
            "timezone": DEFAULT_TZ,
            "images": {
                "placeholder_image": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?q=80&w=1200"
            },
            "rules": {
                "telegram_lookback_hours": 24,
                "redact_fundraising_keywords": FUNDRAISING_KEYWORDS,
            },
            "telegram": {
                "fact_summary": [
                    "https://rsshub.app/telegram/channel/ShrikeNews",
                    "https://rsshub.app/telegram/channel/Yzheleznyak",
                    "https://rsshub.app/telegram/channel/Zvizdecmanhustu",
                ],
                "full_no_opinion": [
                    "https://rsshub.app/telegram/channel/Mccartneyser68"
                ],
                "raw": [
                    "https://rsshub.app/telegram/channel/babchenko77",
                    "https://rsshub.app/telegram/channel/resurgammmm"
                ]
            },
            "weekly_topics": {
                "enabled": True,
                "lookback_days": 7,
                "max_items": 7,
                "per_source_limit": 3,
                "schedule": {
                    "Monday": "archaeology",
                    "Tuesday": "scientific",
                    "Wednesday": "ai",
                    "Thursday": "space",
                    "Friday": "medicine",
                    "Saturday": "energy",
                    "Sunday": "history"
                },
                "sources": {
                    "archaeology": [
                        "https://www.archaeology.org/rss",
                        "https://www.sciencedaily.com/rss/most_recent.xml?topic=archaeology",
                        "https://www.smithsonianmag.com/rss/archaeology/",
                        "https://www.nature.com/subjects/archaeology.rss",
                    ],
                    "scientific": [
                        "https://www.sciencedaily.com/rss/all.xml",
                        "https://www.nature.com/nature/articles?type=news-and-views.rss",
                        "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
                        "https://www.esa.int/rssfeed/Our_Activities/Space_Science",
                    ],
                    "ai": [
                        "https://www.nature.com/subjects/artificial-intelligence.rss",
                        "https://www.technologyreview.com/feed/ai/",
                        "https://arxiv.org/rss/cs.AI",
                        "https://ai.googleblog.com/feeds/posts/default",
                        "https://openai.com/blog/rss",
                    ],
                    "space": [
                        "https://www.nasa.gov/rss/dyn/breaking_news.rss",
                        "https://www.esa.int/rssfeed/Our_Activities/Launchers",
                        "https://www.jpl.nasa.gov/feeds/news",
                        "https://www.space.com/feeds/all",
                    ],
                    "medicine": [
                        "https://www.nih.gov/news-events/news-releases/feed",
                        "https://www.cdc.gov/media/rss.htm",
                        "https://www.sciencedaily.com/rss/health_medicine.xml",
                    ],
                    "energy": [
                        "https://www.iea.org/news.rss",
                        "https://www.noaa.gov/tags/climate/feed",
                        "https://www.sciencedaily.com/rss/earth_climate.xml",
                        "https://www.sciencedaily.com/rss/matter_energy.xml",
                    ],
                    "history": [
                        "https://www.smithsonianmag.com/rss/history/",
                        "https://www.history.com/.rss/full/",
                        "https://www.britishmuseum.org/rss.xml",
                    ],
                }
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gather_tg(cfg):
    """Сбор постов из Telegram-каналов (через RSSHub) по трём режимам."""
    tzname = cfg.get("timezone") or DEFAULT_TZ
    lookback_h = int((cfg.get("rules") or {}).get("telegram_lookback_hours", 24))
    since_utc = datetime.now(timezone.utc) - timedelta(hours=lookback_h)

    out = {"fact_summary": [], "full_no_opinion": [], "raw": []}
    by_channel_count = defaultdict(int)

    for mode in ("fact_summary", "full_no_opinion", "raw"):
        urls = (cfg.get("telegram") or {}).get(mode, [])
        for url_or_list in urls:
            candidates = url_or_list if isinstance(url_or_list, list) else [url_or_list]
            feed = None
            url = None
            for u in candidates:
                try:
                    f = fetch_rss(u)
                    entries = getattr(f, "entries", []) or []
                    if entries:
                        feed = f
                        url = u
                        break
                except Exception as e:
                    print(f"[fetch_rss] skip candidate {u}: {e}")
                    continue
            if feed is None:
                continue

            entries = getattr(feed, "entries", []) or []
            for e in entries:
                pub = parse_time(e)
                if pub < since_utc:
                    continue
                title = getattr(e, "title", "") or ""
                link = getattr(e, "link", "") or ""
                if is_russia_affiliated(link):
                    continue

                # Текст: берём summary/detail и редактируем донаты
                summary_html = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                text = strip_html(summary_html)
                text = redact_fundraising_text(text)

                # Имя канала пытаемся вытащить из URL либо из title
                ch = ""
                m = re.search(r"t\.me/([^/\s]+)", link)
                if m:
                    ch = m.group(1)
                if not ch:
                    # грубый fallback — по домену/заголовку
                    ch = "channel"

                out[mode].append({
                    "channel": ch,
                    "title": title.strip()[:300],
                    "summary_text": text.strip(),
                    "published": pub.isoformat(),
                    "link": link,
                })
                by_channel_count[ch] += 1

    total = sum(len(v) for v in out.values())
    print(f"[gather_tg] fact_summary: {len(out['fact_summary'])} "
          f"full_no_opinion: {len(out['full_no_opinion'])} raw: {len(out['raw'])} TOTAL: {total}")
    print(f"[gather_tg] by channel: {dict(by_channel_count)}")
    return out


# -----------------------------
# Weekly themed digest
# -----------------------------

def gather_weekly_topic(cfg):
    wt = (cfg.get("weekly_topics") or {})
    if not wt.get("enabled"):
        return None

    tzname = cfg.get("timezone", DEFAULT_TZ)
    today_local = now_local(tzname)
    weekday = today_local.strftime("%A")  # Monday..Sunday

    topic_key = (wt.get("schedule") or {}).get(weekday)
    if not topic_key:
        return None

    lookback_days = int(wt.get("lookback_days", 7))
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    max_items = int(wt.get("max_items", 7))
    per_source_limit = int(wt.get("per_source_limit", 3))
    sources = (wt.get("sources") or {}).get(topic_key, [])

    picked = []
    seen = set()
    per_src = defaultdict(int)

    for url in sources:
        try:
            feed = fetch_rss(url)
        except Exception as e:
            print(f"[weekly] skip source {url}: {e}")
            continue
        entries = getattr(feed, "entries", []) or []
        for e in entries:
            pub = parse_time(e)
            if pub < since:
                continue
            link = getattr(e, "link", "") or ""
            if not link or is_russia_affiliated(link):
                continue
            title = getattr(e, "title", "") or ""
            desc_html = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            desc = strip_html(desc_html)

            key = (link.strip(), title.strip()[:140])
            if key in seen:
                continue
            seen.add(key)

            host = re.sub(r"^https?://", "", link).split("/")[0].lower()
            per_src[host] += 1
            if per_src[host] > per_source_limit:
                continue

            picked.append({
                "title": title.strip(),
                "link": link.strip(),
                "published": pub,
                "source": host,
                "summary": desc[:600] + ("…" if len(desc) > 600 else "")
            })

    if not picked:
        return None

    picked.sort(key=lambda x: x["published"], reverse=True)
    picked = picked[:max_items]
    local_dt = picked[0]["published"].astimezone(pytz.timezone(tzname))
    local_str = local_dt.strftime("%Y-%m-%d %H:%M")
    title = f"Weekly Digest • {topic_key.capitalize()} — {local_dt.strftime('%Y-%m-%d')}"

    bullets = []
    for it in picked:
        date_str = it["published"].astimezone(pytz.timezone(tzname)).strftime("%Y-%m-%d")
        bullets.append(
            f"• <b>{html.escape(it['title'])}</b> (<i>{html.escape(it['source'])}</i>, {date_str}) — "
            f"{html.escape(it['summary'])} <br/><a href=\"{it['link']}\">{it['link']}</a>"
        )

    description_html = f"{local_str} — подборка {len(picked)} материалов:<br/><br/>" + "<br/>".join(bullets)
    placeholder_image = (cfg.get("images") or {}).get("placeholder_image")

    return {
        "category": f"weekly:{topic_key}",
        "title": title,
        "link": "https://magetarro.github.io/news/rss.xml",
        "published": picked[0]["published"].isoformat(),
        "description_html": description_html,
        "image": placeholder_image
    }


def render_weekly_item(item, tzname):
    if not item:
        return ""
    dt = dtparser.parse(item["published"])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (
        "    <item>\n"
        f"      <title>{item['title']}</title>\n"
        f"      <link>{item['link']}</link>\n"
        f"      <guid isPermaLink=\"false\">urn:weekly:{int(dt.timestamp())}</guid>\n"
        f"      <category>{item['category']}</category>\n"
        f"      <pubDate>{to_rfc822(item['published'])}</pubDate>\n"
        f"      <description><![CDATA[{item.get('description_html','')}]]></description>\n"
        f"      <enclosure url=\"{item.get('image','')}\" type=\"image/jpeg\" length=\"0\" />\n"
        "    </item>\n"
    )


# -----------------------------
# OpenAI call (small payloads, retries)
# -----------------------------

def openai_chat(messages, model=None):
    """Мини-клиент Chat Completions с ретраями и увеличенным timeout."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = model or OPENAI_MODEL
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    delay = 1.2
    for attempt in range(1, OPENAI_RETRIES + 1):
        try:
            # (connect, read) timeouts
            resp = requests.post(url, headers=headers, json=payload, timeout=(10, 120))
            if resp.status_code != 200:
                print("OpenAI API error", resp.status_code, resp.text[:500], file=sys.stderr)
                # 429/5xx — попробуем ещё раз
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < OPENAI_RETRIES:
                    print(f"[openai_chat] retry in {round(delay,1)}s after HTTP {resp.status_code}")
                    time.sleep(delay)
                    delay *= OPENAI_BACKOFF
                    continue
                raise RuntimeError(f"OpenAI HTTP {resp.status_code}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < OPENAI_RETRIES:
                print(f"[openai_chat] retry in {round(delay,1)}s after error: {e}")
                time.sleep(delay)
                delay *= OPENAI_BACKOFF
                continue
            print(f"[WARN] OpenAI call failed, using fallback: {e}")
            raise


# -----------------------------
# Channel-wise building (small chunks)
# -----------------------------

def compact_items(items, max_chars=800):
    seen = set()
    out = []
    for it in items:
        key = (it.get("link","").strip(), (it.get("title") or "")[:120], it.get("channel"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "channel": it.get("channel"),
            "title": (it.get("title") or "")[:200],
            "text": (it.get("summary_text") or "")[:max_chars],
            "published": it.get("published"),
            "link": it.get("link"),
        })
    return out


def group_by_channel(items):
    g = defaultdict(list)
    for it in items:
        g[it["channel"]].append(it)
    return g


def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


PROMPT_RSS_ITEMS = (
    "Собери RSS <item> элементы из входных данных. Для КАЖДОГО объекта верни ОДИН <item>:\n"
    "- <title>: краткий факт-заголовок (если пусто — сгенерируй по содержанию, без мнений)\n"
    "- <link>: исходная ссылка\n"
    "- <category>: имя канала (поле 'channel')\n"
    "- <pubDate>: RFC822 из поля 'published'\n"
    "- <description>: 3–5 предложений фактов (RU/UK), без мнений/оценок.\n"
    "Верни ТОЛЬКО последовательность <item>...</item>, без <rss> и <channel>."
)


def llm_items_for_chunk(channel, items, model):
    data = {"channel": channel, "items": items}
    messages = [
        {"role": "system", "content": "Ты формируешь компактные RSS <item> элементы для дайджеста."},
        {"role": "user", "content": PROMPT_RSS_ITEMS},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False)}
    ]
    return openai_chat(messages, model=model)


def render_minimal_items(items, tzname):
    res = []
    tz = pytz.timezone(tzname)
    for it in items:
        dt = dtparser.parse(it["published"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_str = dt.astimezone(tz).strftime("%Y-%m-%d %H:%M")
        title = it.get("title") or f"Пост @{it.get('channel')}"
        link = it.get("link") or ""
        desc = it.get("text") or ""
        xml = (
            "    <item>\n"
            f"      <title>{html.escape(title)}</title>\n"
            f"      <link>{link}</link>\n"
            f"      <guid isPermaLink=\"false\">urn:fallback:{it.get('channel')}:{int(dt.timestamp())}</guid>\n"
            f"      <category>{html.escape(it.get('channel') or '')}</category>\n"
            f"      <pubDate>{to_rfc822(it['published'])}</pubDate>\n"
            f"      <description><![CDATA[{local_str} — {html.escape(desc)}]]></description>\n"
            "    </item>\n"
        )
        res.append(xml)
    return "\n".join(res)


def build_items_channelwise(tg_trimmed, model, tzname, per_chunk=6):
    all_items_xml = []
    for mode in ("fact_summary", "full_no_opinion", "raw"):
        by_ch = group_by_channel(tg_trimmed.get(mode, []))
        for ch, posts in by_ch.items():
            posts_sorted = sorted(posts, key=lambda x: x["published"], reverse=True)
            for part in chunk(posts_sorted, per_chunk):
                try:
                    xml_part = llm_items_for_chunk(ch, part, model)
                    all_items_xml.append(xml_part.strip())
                except Exception as e:
                    print(f"[llm] chunk failed for channel {ch}: {e}")
                    all_items_xml.append(render_minimal_items(part, tzname))
                time.sleep(0.4)
    return "\n".join(all_items_xml)


def wrap_items_into_rss(channel_title, items_xml, tzname):
    now = datetime.now(pytz.timezone(tzname))
    head = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<rss version=\"2.0\">\n"
        "  <channel>\n"
        f"    <title>{channel_title}</title>\n"
        "    <link>https://magetarro.github.io/news/rss.xml</link>\n"
        "    <description>Сводный Telegram-дайджест с тематическими материалами.</description>\n"
        "    <language>ru</language>\n"
        f"    <lastBuildDate>{now.strftime('%a, %d %b %Y %H:%M:%S %z')}</lastBuildDate>\n"
    )
    tail = "  </channel>\n</rss>\n"
    return head + (items_xml or "") + "\n" + tail


# -----------------------------
# Fallback deterministic builder (no LLM)
# -----------------------------

def render_items_from_mode(items, category, tzname, placeholder_image=None):
    out = []
    for it in items:
        title = it.get("title") or f"Пост @{it.get('channel')}"
        link = it.get("link") or ""
        pub = it.get("published") or datetime.now(timezone.utc).isoformat()
        desc = it.get("summary_text") or ""
        xml = (
            "    <item>\n"
            f"      <title>{html.escape(title)}</title>\n"
            f"      <link>{link}</link>\n"
            f"      <guid isPermaLink=\"false\">urn:{category}:{hashlib.md5((link+pub).encode()).hexdigest()}</guid>\n"
            f"      <category>{category}</category>\n"
            f"      <pubDate>{to_rfc822(pub)}</pubDate>\n"
            f"      <description><![CDATA[{html.escape(desc)}]]></description>\n"
        )
        if placeholder_image:
            xml += f"      <enclosure url=\"{placeholder_image}\" type=\"image/jpeg\" length=\"0\" />\n"
        xml += "    </item>\n"
        out.append(xml)
    return out


def build_rss_fallback_from_tg(tg, cfg, weekly_item=None):
    tzname = cfg.get("timezone") or DEFAULT_TZ
    now = datetime.now(pytz.timezone(tzname))
    placeholder = (cfg.get("images") or {}).get("placeholder_image")

    head = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<rss version=\"2.0\">\n"
        "  <channel>\n"
        "    <title>Telegram дайджест (fallback)</title>\n"
        "    <link>https://magetarro.github.io/news/rss.xml</link>\n"
        "    <description>Резервная сборка без LLM</description>\n"
        "    <language>ru</language>\n"
        f"    <lastBuildDate>{now.strftime('%a, %d %b %Y %H:%M:%S %z')}</lastBuildDate>\n"
    )

    items_xml = []
    if weekly_item:
        items_xml.append(render_weekly_item(weekly_item, tzname))
    items_xml += render_items_from_mode(tg.get("fact_summary", []), "fact_summary", tzname, placeholder)
    items_xml += render_items_from_mode(tg.get("full_no_opinion", []), "full_no_opinion", tzname, placeholder)
    items_xml += render_items_from_mode(tg.get("raw", []), "raw", tzname, placeholder)
    tail = "  </channel>\n</rss>\n"
    return head + "".join(items_xml) + tail


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"[env] model={OPENAI_MODEL} ref={os.getenv('GITHUB_REF','(local)')}")

    cfg = load_config()
    tg = gather_tg(cfg)

    # Отсечём потолки, чтобы не раздувать payload
    def cap_per_channel(items, n=MAX_PER_CHANNEL):
        by = group_by_channel(items)
        out = []
        for ch, posts in by.items():
            posts = sorted(posts, key=lambda x: x["published"], reverse=True)[:n]
            out += posts
        return out

    tg = {
        "fact_summary": sorted(tg.get("fact_summary", []), key=lambda x: x["published"], reverse=True)[:MAX_PER_MODE],
        "full_no_opinion": sorted(tg.get("full_no_opinion", []), key=lambda x: x["published"], reverse=True)[:MAX_PER_MODE],
        "raw": sorted(tg.get("raw", []), key=lambda x: x["published"], reverse=True)[:MAX_PER_MODE],
    }
    tg = {k: cap_per_channel(v, MAX_PER_CHANNEL) for k, v in tg.items()}

    # компакт для LLM
    tg_trimmed = {
        "fact_summary": compact_items(tg.get("fact_summary", []), max_chars=MAX_CHARS_PER_ITEM),
        "full_no_opinion": compact_items(tg.get("full_no_opinion", []), max_chars=MAX_CHARS_PER_ITEM),
        "raw": compact_items(tg.get("raw", []), max_chars=MAX_CHARS_PER_ITEM),
    }
    items_total = sum(len(v) for v in tg_trimmed.values())
    print(f"[DEBUG] prepared for LLM: {items_total} items "
          f"(chars/item≤{MAX_CHARS_PER_ITEM}, per_channel≤{MAX_PER_CHANNEL}, per_chunk={LLM_PER_CHUNK})")

    # weekly-item
    weekly = gather_weekly_topic(cfg)
    if weekly:
        print(f"[weekly] added topic item: {weekly['category']} — {weekly['title']}")
    else:
        print("[weekly] no topic item today or nothing picked.")

    # Пытаемся собрать каналами (если OpenAI откажет — fallback ниже)
    try:
        items_xml = build_items_channelwise(tg_trimmed, OPENAI_MODEL, cfg.get("timezone", DEFAULT_TZ), per_chunk=LLM_PER_CHUNK)
        weekly_xml = render_weekly_item(weekly, cfg.get("timezone", DEFAULT_TZ)) if weekly else ""
        final_xml = wrap_items_into_rss("Telegram дайджест (канально)", weekly_xml + "\n" + items_xml, cfg.get("timezone", DEFAULT_TZ))
        # валидация
        etree.fromstring(final_xml.encode("utf-8"))
        rss_xml = final_xml
    except Exception as e:
        print(f"[WARN] channelwise path failed, fallback to deterministic builder: {e}")
        rss_xml = build_rss_fallback_from_tg(tg, cfg, weekly_item=weekly)

    with open("rss.xml", "w", encoding="utf-8") as f:
        f.write(rss_xml)
    print("rss.xml written.")


if __name__ == "__main__":
    main()
