
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, re, json, requests, feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import pytz
from lxml import etree
import yaml

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY secret.", file=sys.stderr)
    sys.exit(1)

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # optional
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")            # optional

CONFIG_PATH = "sources.yml"

def load_cfg(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def now_local(tzname):
    return datetime.now(pytz.timezone(tzname))

def should_publish(now_local_dt, publish_hour):
    return now_local_dt.hour == publish_hour

def fetch_rss(url):
    return feedparser.parse(url)

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

def extract_first_image_from_entry(entry, summary_html):
    # Try standard fields in feedparser first
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

    # Fallback: parse <img src="..."> from summary HTML
    if summary_html:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary_html, flags=re.I)
        if m:
            return m.group(1)
    return None

def find_external_image(query):
    # Prefer Unsplash if key provided, else Pexels, else None
    query = query.strip()
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
                    # choose 'urls'->'regular' or 'small'
                    url = data["results"][0]["urls"].get("regular") or data["results"][0]["urls"].get("small")
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
                    return url
        except Exception:
            pass

    return None

def contains_keywords(text, keywords):
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in keywords if k)

def gather_tg(cfg):
    lookback_h = int(cfg["rules"].get("telegram_lookback_hours", 24))
    since_utc = datetime.now(timezone.utc) - timedelta(hours=lookback_h)
    kw = [k.lower() for k in cfg["rules"].get("fundraising_filter_keywords", [])]
    placeholder_image = (cfg.get("images") or {}).get("placeholder_image")
    enable_external = (cfg.get("images") or {}).get("enable_external_images", True)

    out = {"fact_summary": [], "full_no_opinion": [], "raw": []}
    for mode in ("fact_summary", "full_no_opinion", "raw"):
        for url in cfg["telegram"].get(mode, []):
            feed = fetch_rss(url)
            for e in feed.entries:
                pub = parse_time(e)
                if pub < since_utc:
                    continue
                title = getattr(e, "title", "") or ""
                link = getattr(e, "link", url)
                summary_html = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                txt = strip_html(summary_html + " " + title)

                # fundraising exclusion (RU + UK + EN)
                if contains_keywords(txt, kw):
                    continue

                # channel name from URL
                m = re.search(r"/telegram/channel/([^/?#]+)", url)
                channel = m.group(1) if m else url

                # Choose image: first from post, else external, else placeholder
                image_url = extract_first_image_from_entry(e, summary_html)
                if not image_url and enable_external:
                    # Build a short query: channel + trimmed title (or first 6 words of text)
                    base_query = title.strip() or "новости пост телеграм"
                    if not base_query:
                        base_query = "новости"
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
    return out

def openai_chat(messages, model="gpt-5-thinking", temperature=0):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    if resp.status_code != 200:
        print("OpenAI API error", resp.status_code, resp.text[:500], file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def validate_xml(xml_text):
    try:
        etree.fromstring(xml_text.encode("utf-8"))
        return True, ""
    except Exception as ex:
        return False, str(ex)

def main():
    cfg = load_cfg()
    tz = cfg["timezone"]
    hour = int(cfg["publish_hour_local"])
    nloc = now_local(tz)
    if not should_publish(nloc, hour):
        print(f"Not publish window: {nloc.isoformat()}")
        sys.exit(0)

    tg = gather_tg(cfg)

    opinion_markers = [
        "я думаю","по моему мнению","мне кажется","я считаю","как по мне","на мой взгляд","по-моему","лично я","я уверен","думаю",
        "я думаю","на мою думку","мені здається","я вважаю","як на мене","на мій погляд","по-моєму","особисто я","я впевнений","гадаю",
        "i think","in my opinion","i believe","personally","seems to me"
    ]

    rules_text = (
        "Сформируй один RSS 2.0 (один канал). Режимы:\n"
        "- fact_summary: краткое фактическое резюме 3–5 предложений (на языке поста RU/UK), только факты.\n"
        "- full_no_opinion: полный текст (RU/UK), но удали предложения с личными оценками (маркеры: "
        + "; ".join(opinion_markers) + "). Сохрани факты/цифры/даты/цитаты.\n"
        "- raw: без изменений. Если передан image — добавь <enclosure>.\n"
        "Для каждого item добавляй: <title>, <category> с именем канала, локальное время America/Los_Angeles (YYYY-MM-DD HH:MM) в начале description, <link>, затем содержимое по режиму, и <enclosure> с image, если он есть.\n"
        "Верни только валидный XML, начиная с <?xml ...>, без пояснений."
    )

    input_json = {
        "output": {
            "type": "rss2.0",
            "channel": {
                "title": "Telegram дайджест (с изображениями)",
                "link": "https://magetarro.github.io/news/rss.xml",
                "description": "Сводный Telegram-дайджест: fact summary, full без мнений, raw; с картинками (первая из поста, иначе подобранная/заглушка).",
                "language": "ru"
            }
        },
        "timezone": cfg["timezone"],
        "data": tg
    }

    messages = [
        {"role": "system", "content": "Ты помощник, который строит RSS 2.0 из входных данных по чётким правилам."},
        {"role": "user", "content": rules_text},
        {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)}
    ]

    rss_xml = openai_chat(messages, model="gpt-5-thinking", temperature=0)

    ok, err = validate_xml(rss_xml)
    if not ok:
        print("XML validation failed:", err, file=sys.stderr)
        m = re.search(r"(<\?xml[\s\S]+</rss>)", rss_xml)
        if m:
            rss_xml = m.group(1)
            ok, err = validate_xml(rss_xml)
    if not ok:
        print("Final XML still invalid.", file=sys.stderr)
        sys.exit(1)

    with open("rss.xml", "w", encoding="utf-8") as f:
        f.write(rss_xml)
    print("rss.xml written.")

if __name__ == "__main__":
    main()
