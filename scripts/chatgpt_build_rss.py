
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

def contains_keywords(text, keywords):
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in keywords if k)

def gather_tg(cfg):
    lookback_h = int(cfg["rules"].get("telegram_lookback_hours", 24))
    since_utc = datetime.now(timezone.utc) - timedelta(hours=lookback_h)
    kw = [k.lower() for k in cfg["rules"].get("fundraising_filter_keywords", [])]

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

                out[mode].append({
                    "channel": channel,
                    "title": title.strip() or f"Пост @{channel}",
                    "link": link,
                    "summary_html": summary_html,
                    "summary_text": txt,
                    "published": pub.isoformat()
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

    # Opinion markers for RU + UK (used by ChatGPT to remove subjective sentences in full_no_opinion mode)
    opinion_markers = [
        # Russian
        "я думаю", "по моему мнению", "мне кажется", "я считаю", "как по мне",
        "на мой взгляд", "по-моему", "лично я", "я уверен", "думаю",
        # Ukrainian
        "я думаю", "на мою думку", "мені здається", "я вважаю", "як на мене",
        "на мій погляд", "по-моєму", "особисто я", "я впевнений", "гадаю",
        # English common
        "i think", "in my opinion", "i believe", "personally", "seems to me"
    ]

    rules_text = (
        "Сформируй один RSS 2.0 (один канал). Для Telegram-постов действуют режимы:\n"
        "- fact_summary: для каждого поста сделай краткое фактическое резюме 3–5 предложений на русском или украинском (в зависимости от языка поста). Включай даты, цифры, имена, географию; полностью убери личные оценки/мнения автора.\n"
        "- full_no_opinion: выведи полный текст поста (рус./укр.), но удали предложения с личными оценками/мнениями. Считай субъективными любые фразы с маркерами: "
        + "; ".join(opinion_markers) + ". Сохрани факты, цитаты и нейтральные формулировки.\n"
        "- raw: без изменений; можно минимально очистить HTML.\n"
        "Для каждого item добавляй: <title> (оригинал или первая строка), <category> с именем канала, локальное время публикации America/Los_Angeles (YYYY-MM-DD HH:MM) в описании либо начале текста, <link>, <description> по режиму. Если доступно изображение — добавь <enclosure> с ссылкой.\n"
        "Верни только валидный XML, начиная с <?xml ...>, без пояснений."
    )

    input_json = {
        "output": {
            "type": "rss2.0",
            "channel": {
                "title": "Telegram дайджест (RU+UK правила)",
                "link": "https://magetarro.github.io/news/rss.xml",
                "description": "Сводный Telegram-дайджест с правилами для RU/UK: fact summary, full без мнений, и raw.",
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
