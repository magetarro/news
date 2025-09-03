#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, random, requests, feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import pytz
from lxml import etree
import yaml
from collections import Counter

# ---- ENV / CONFIG ----
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY secret.", file=sys.stderr)
    sys.exit(1)

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # optional
PEXELS_API_KEY     = os.getenv("PEXELS_API_KEY")        # optional
OPENAI_MODEL       = os.getenv("OPENAI_MODEL") or "pt-5-mini"  # можно: gpt-5, gpt-5-mini, gpt-4.1-mini и т.п.

CONFIG_PATH = "sources.yml"


# --------------------- helpers ---------------------

def load_cfg(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def now_local(tzname):
    return datetime.now(pytz.timezone(tzname))

def should_publish(now_local_dt, publish_hour):
    return now_local_dt.hour == publish_hour

def fetch_rss(url):
    """Fetch RSS with retries/backoff; always returns a feed-like object with entries list."""
    tries = 4
    base = 2.0
    empty_feed = feedparser.parse(b"")  # has .entries == []
    for i in range(tries):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "magetarro-rss-bot/1.1 (+https://github.com/magetarro/news)"},
                timeout=(10, 25),  # connect, read
            )
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                delay = float(ra) if (ra and ra.isdigit()) else (base * (2 ** i))
                delay += random.uniform(0, 0.9)
                print(f"[fetch_rss] 429, sleep {delay:.1f}s {url}")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            feed = feedparser.parse(resp.content) or empty_feed
            print(f"[fetch_rss] OK {resp.status_code} {url} entries={len(feed.entries)}")
            return feed
        except Exception as e:
            if i < tries - 1:
                delay = base * (2 ** i) + random.uniform(0, 0.9)
                print(f"[fetch_rss] error, retry in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                print(f"[fetch_rss] FAIL {url}: {e}")
                return empty_feed
    return empty_feed


def parse_time(entry):
    pub = None
    for key in ("published", "updated", "created"):
        val = getattr(entry, key, None)
        if val:
            try:
                pub = dtparser.parse(val)
                break
            except Exception:
                pass
    if pub is None:
        pub = datetime.now(timezone.utc)
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return pub

def strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_keywords(text, keywords):
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in keywords if k)

def extract_first_image_from_entry(entry, summary_html):
    # Try standard media fields
    if getattr(entry, "media_content", None):
        for m in entry.media_content:
            url = m.get("url")
            if url:
                return url
    if getattr(entry, "media_thumbnail", None):
        for m in entry.media_thumbnail:
            url = m.get("url")
            if url:
                return url
    if getattr(entry, "enclosures", None):
        for enc in entry.enclosures:
            if isinstance(enc, dict) and enc.get("href", "").startswith("http"):
                return enc["href"]
    # Parse <img> from HTML
    if summary_html:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary_html, flags=re.I)
        if m:
            return m.group(1)
    return None

def find_external_image(query):
    """Optional Unsplash/Pexels lookup. Returns URL or None."""
    query = (query or "").strip()
    if not query:
        return None

    if UNSPLASH_ACCESS_KEY:
        try:
            resp = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": 1},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    url = data["results"][0]["urls"].get("regular") or data["results"][0]["urls"].get("small")
                    if url:
                        return url
        except Exception:
            pass

    if PEXELS_API_KEY:
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": 1},
                headers={"Authorization": PEXELS_API_KEY},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                photos = data.get("photos") or []
                if photos:
                    src = photos[0].get("src", {})
                    url = src.get("large") or src.get("medium") or src.get("original")
                    if url:
                        return url
        except Exception:
            pass

    return None

def redact_fundraising(text, keywords):
    """
    Удаляет из текста предложения/строки, где встречаются маркеры донатов/реквизитов.
    Дополнительно вырезает типовые паттерны ссылок/кошельков/криптокошельков.
    """
    if not text:
        return text

    low_markers = [k.lower() for k in keywords if k]
    # Разбиваем аккуратно по предложениям/строкам
    parts = re.split(r'(?<=[.!?…])\s+|\n+', text)
    keep = []
    for p in parts:
        pl = p.lower()
        if any(k in pl for k in low_markers):
            continue
        keep.append(p)

    redacted = " ".join(keep).strip()

    # Жёсткое вычищение явных реквизитов/кошельков/ссылок
    patterns = [
        r'(https?://)?(www\.)?(patreon|boosty|buymeacoffee|paypal\.me)/\S+',
        r'(btc|eth|usdt|xmr):\S+',
        r'(кошел[её]к|wallet)\s*[:：]\s*\S+',
        r'(card|карт[аы]|картку|карта)\s*[:：]?\s*\d[\d\s\-]{8,}',  # номера карт
        r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # btc-like
    ]
    for pat in patterns:
        redacted = re.sub(pat, '〔удалено〕', redacted, flags=re.I)

    # Сжимаем лишние пробелы
    redacted = re.sub(r'\s{2,}', ' ', redacted).strip()
    return redacted


# --------------------- data gathering ---------------------

def gather_tg(cfg):
    lookback_h = int(cfg["rules"].get("telegram_lookback_hours", 24))
    since_utc = datetime.now(timezone.utc) - timedelta(hours=lookback_h)

    redact_kw = [k.lower() for k in (
        cfg["rules"].get("redact_fundraising_keywords") or
        cfg["rules"].get("fundraising_filter_keywords") or []
    )]

    placeholder_image = (cfg.get("images") or {}).get("placeholder_image")
    enable_external   = (cfg.get("images") or {}).get("enable_external_images", True)

    out = {"fact_summary": [], "full_no_opinion": [], "raw": []}

    for mode in ("fact_summary", "full_no_opinion", "raw"):
        for url in cfg["telegram"].get(mode, []):
            feed = fetch_rss(url)
            entries = getattr(feed, "entries", []) or []
            for e in entries:
                pub = parse_time(e)
                if pub < since_utc:
                    continue

                title = getattr(e, "title", "") or ""
                link = getattr(e, "link", url)
                summary_html = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                txt = strip_html(summary_html + " " + title)

                # Редактируем, убираем донаты
                txt = redact_fundraising(txt, redact_kw)

                m = re.search(r"/telegram/channel/([^/?#]+)", url)
                channel = m.group(1) if m else url

                image_url = extract_first_image_from_entry(e, summary_html)
                if not image_url and enable_external:
                    base_query = title.strip() or "новости пост телеграм"
                    image_url = find_external_image(f"{channel} {base_query}")
                if not image_url:
                    image_url = placeholder_image

                out[mode].append({
                    "channel": channel,
                    "title": title.strip() or f"Пост @{channel}",
                    "link": link,
                    "summary_html": summary_html,
                    "summary_text": txt,
                    "published": pub.isoformat(),
                    "image": image_url,
                })
            # чуть притормозим между каналами, чтобы не ловить 429
            time.sleep(0.4 + random.uniform(0, 0.4))

    total = sum(len(out[k]) for k in out)
    print("[gather_tg] fact_summary:", len(out["fact_summary"]),
          "full_no_opinion:", len(out["full_no_opinion"]),
          "raw:", len(out["raw"]), "TOTAL:", total)
    by_channel = Counter([it["channel"] for mode in out for it in out[mode]])
    print("[gather_tg] by channel:", dict(by_channel))
    return out

# --------------------- OpenAI call ---------------------

def openai_chat(messages, model=None):
    """OpenAI call with retries/backoff; no temperature in payload."""
    model = model or OPENAI_MODEL
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    # optional GPT-5 family params via ENV
    v = os.getenv("OPENAI_VERBOSITY")
    r = os.getenv("OPENAI_REASONING_EFFORT")
    if v: payload["verbosity"] = v
    if r: payload["reasoning_effort"] = r

    tries = int(os.getenv("OPENAI_RETRIES", "3"))
    base = float(os.getenv("OPENAI_BACKOFF", "2.0"))
    for i in range(tries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(10, 60))  # connect, read
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            if 500 <= resp.status_code < 600:
                raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
        except Exception as e:
            if i == tries - 1:
                raise
            delay = base * (2 ** i) + random.uniform(0, 0.9)
            print(f"[openai_chat] retry in {delay:.1f}s after error: {e}")
            time.sleep(delay)


# --------------------- fallback builder ---------------------

def count_items(xml_text: str) -> int:
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
        return len(root.xpath("//channel/item"))
    except Exception:
        return 0

def split_sentences(text, max_sentences=5):
    parts = re.split(r'(?<=[.!?…])\s+', (text or "").strip())
    parts = [p for p in parts if p]
    return " ".join(parts[:max_sentences]) if parts else text

def remove_opinion_sentences(text):
    markers = [
        # RU
        "я думаю","по моему мнению","мне кажется","я считаю","как по мне",
        "на мой взгляд","по-моему","лично я","думаю","я уверен","убеждён",
        # UK
        "на мою думку","мені здається","я вважаю","як на мене","на мій погляд","особисто я",
        # EN
        "i think","in my opinion","i believe","personally","seems to me",
    ]
    sents = re.split(r'(?<=[.!?…])\s+', (text or "").strip())
    keep = []
    low_markers = [m.lower() for m in markers]
    for s in sents:
        if not s.strip():
            continue
        if any(m in s.lower() for m in low_markers):
            continue
        keep.append(s)
    return " ".join(keep).strip() or (text or "")

def to_rfc822(dt_iso):
    dt = dtparser.parse(dt_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S %z")

def render_items_from_mode(items, mode, tzname, placeholder_image):
    res = []
    tz = pytz.timezone(tzname)
    for it in items:
        dt = dtparser.parse(it["published"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(tz)
        local_str = local_dt.strftime("%Y-%m-%d %H:%M")

        title = it["title"] or f"Пост @{it['channel']}"
        link = it["link"]
        body = it.get("summary_text") or ""
        image = it.get("image") or placeholder_image

        if mode == "fact_summary":
            desc = split_sentences(body, max_sentences=5)
        elif mode == "full_no_opinion":
            desc = remove_opinion_sentences(body)
        else:
            desc = body

        description = f"{local_str} — {desc}"
        item_xml = (
            "    <item>\n"
            f"      <title>{title}</title>\n"
            f"      <link>{link}</link>\n"
            f"      <guid isPermaLink=\"false\">urn:tg:{it['channel']}:{int(dt.timestamp())}</guid>\n"
            f"      <category>{it['channel']}</category>\n"
            f"      <pubDate>{to_rfc822(it['published'])}</pubDate>\n"
            f"      <description><![CDATA[{description}]]></description>\n"
            f"      <enclosure url=\"{image}\" type=\"image/jpeg\" length=\"0\" />\n"
            "    </item>\n"
        )
        res.append(item_xml)
    return res

def build_rss_fallback_from_tg(tg, cfg):
    tzname = cfg["timezone"]
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
    items_xml += render_items_from_mode(tg.get("fact_summary", []), "fact_summary", tzname, placeholder)
    items_xml += render_items_from_mode(tg.get("full_no_opinion", []), "full_no_opinion", tzname, placeholder)
    items_xml += render_items_from_mode(tg.get("raw", []), "raw", tzname, placeholder)

    tail = "  </channel>\n</rss>\n"
    return head + "".join(items_xml) + tail


# --------------------- main ---------------------

def main():
    print(f"[env] model={OPENAI_MODEL} ref={os.getenv('GITHUB_REF','(local)')}")
    cfg = load_cfg()
    tz = cfg["timezone"]
    hour = int(cfg["publish_hour_local"])
    nloc = now_local(tz)

    # Minute-window gate (опционально ужесточите по желанию)
    force = os.getenv("FORCE_PUBLISH", "false").lower() in ("1","true","yes","on")
    within_window = (nloc.hour == hour)  # можно сузить до минут: and 58 <= nloc.minute <= 3
    if not (force or within_window):
        print(f"Skip: local={nloc.strftime('%H:%M')}, want ~{hour:02d}:00")
        sys.exit(0)

    tg = gather_tg(cfg)
    total = sum(len(tg[k]) for k in tg)
    print("[DEBUG] collected:", total, "items")
    for mode, items in tg.items():
        print("  ", mode, "->", len(items))
        for it in items[:3]:
            print("    ", it["channel"], "|", it["title"][:80])

    if total == 0:
        raise RuntimeError("No Telegram posts collected in lookback window. "
                           "Check RSSHub availability, filters, and lookback hours.")

    # ---------- Prepare trimmed data for LLM (speed) ----------
    MAX_PER_MODE = int(os.getenv("MAX_PER_MODE", "6"))  # LLM limit per mode
    def trim(items): 
        return sorted(items, key=lambda x: x["published"], reverse=True)[:MAX_PER_MODE]

    tg_trimmed = {
        "fact_summary": trim(tg.get("fact_summary", [])),
        "full_no_opinion": trim(tg.get("full_no_opinion", [])),
        "raw": trim(tg.get("raw", [])),
    }

    # ---------- Build via LLM ----------
    opinion_markers = [
        "я думаю","по моему мнению","мне кажется","я считаю","как по мне",
        "на мой взгляд","по-моему","лично я","думаю","я уверен","убеждён",
        "на мою думку","мені здається","я вважаю","як на мене","на мій погляд","особисто я",
        "i think","in my opinion","i believe","personally","seems to me"
    ]

    rules_text = (
        "Сформируй один RSS 2.0 (один канал). Режимы:\n"
        "- fact_summary: резюме 3–5 фактических предложений (RU/UK). Без оценок.\n"
        "- full_no_opinion: полный текст (RU/UK), но удали субъективные предложения с маркерами: "
        + "; ".join(opinion_markers) + ".\n"
        "- raw: без изменений. Если есть image — добавь <enclosure>.\n"
        "Важно: тексты уже очищены от упоминаний донатов/реквизитов; не возвращай просьбы о пожертвованиях.\n"
        "Для каждого item добавляй: <title>, <category> (имя канала), локальное время America/Los_Angeles (YYYY-MM-DD HH:MM) в начале description, <link>, содержимое по режиму, и <enclosure>.\n"
        "Верни только валидный XML, начиная с <?xml ...>. Если data содержит элементы, ОБЯЗАТЕЛЬНО создай <item> для каждого. Не выпускать пустой канал."
    )

    input_json = {
        "output": {
            "type": "rss2.0",
            "channel": {
                "title": "Telegram дайджест (с изображениями)",
                "link": "https://magetarro.github.io/news/rss.xml",
                "description": "Сводный Telegram-дайджест: fact summary, full без мнений, raw; с картинками.",
                "language": "ru"
            }
        },
        "timezone": cfg["timezone"],
        "data": tg_trimmed
    }

    messages = [
        {"role": "system", "content": "Ты помощник, который строит RSS 2.0 из входных данных по чётким правилам."},
        {"role": "user", "content": rules_text},
        {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)}
    ]

    print("[DEBUG] sending to ChatGPT (trimmed)")
    print(json.dumps(input_json, ensure_ascii=False)[:2000])

    try:
        rss_xml = openai_chat(messages, model=OPENAI_MODEL)
        items_count = count_items(rss_xml)
        print(f"[DEBUG] LLM items count: {items_count}")
    except Exception as e:
        print(f"[WARN] OpenAI call failed, using fallback: {e}")
        rss_xml = build_rss_fallback_from_tg(tg, cfg)
        items_count = count_items(rss_xml)

    # ---------- Fallback если LLM вернул пустое ----------
    if items_count == 0:
        print("[WARN] LLM produced empty RSS. Falling back to deterministic builder.")
        rss_xml = build_rss_fallback_from_tg(tg, cfg)

    # Простая XML-валидация
    try:
        etree.fromstring(rss_xml.encode("utf-8"))
    except Exception as ex:
        print("[WARN] XML validation failed, trying to extract core <rss> block:", ex)
        m = re.search(r"(<\?xml[\s\S]+</rss>)", rss_xml)
        if m:
            rss_xml = m.group(1)
            etree.fromstring(rss_xml.encode("utf-8"))  # если снова упадёт — исключение

    with open("rss.xml", "w", encoding="utf-8") as f:
        f.write(rss_xml)
    print("rss.xml written.")

if __name__ == "__main__":
    main()
