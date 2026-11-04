"""jevner language sweep: end-to-end across Asia/Europe/Americas.

Runs the full pipeline on 23 languages (12 Asian / 9 European / 3
Americas) against the live API; each language must surface at least one
expected entity, and full results are printed for review. Requires
TYPESAFE_API_KEY / TYPESAFE_BASE_URL and bills the API.
"""

from __future__ import annotations

from jevner import JevNER

# (language, continent, test text, expected entity substring)
LANGUAGE_CASES: list[tuple[str, str, str, str]] = [
    # --- Asia ---
    ("Chinese", "Asia", "你好，我的名字是陈嘉禾，来自江苏南京。", "陈嘉禾"),
    ("Japanese", "Asia", "田中さんは東京大学の学生です。", "田中"),
    ("Korean", "Asia", "김철수 씨는 서울에서 삽니다.", "서울"),
    ("Thai", "Asia", "สวัสดีครับ ผมชื่อสมชาย มาจากกรุงเทพมหานคร", "สมชาย"),
    ("Vietnamese", "Asia", "Nguyễn Văn An làm việc tại Hà Nội.", "Hà Nội"),
    ("Indonesian", "Asia", "Presiden Joko Widodo berkunjung ke Jakarta.", "Jakarta"),
    ("Malay", "Asia", "Hai, nama saya Ahmad, saya berasal dari Kuala Lumpur.", "Kuala Lumpur"),
    ("Filipino", "Asia", "Si Maria ay nagtatrabaho sa Maynila.", "Maria"),
    ("Hindi", "Asia", "राम शर्मा दिल्ली में रहता है।", "राम"),
    ("Arabic", "Asia", "مرحبا، اسمي محمد وأعمل في شركة أرامكو في الرياض.", "محمد"),
    ("Hebrew", "Asia", "שלום, שמי דוד ואני גר בירושלים.", "דוד"),
    ("Turkish", "Asia", "Ahmet İstanbul'da bir şirkette çalışıyor.", "Ahmet"),
    # --- Europe ---
    ("English", "Europe", "Hi, i'm Vortez Wohl, working at Google in London.", "Vortez Wohl"),
    ("French", "Europe", "Bonjour, je m'appelle Marie Dubois et j'habite à Paris.", "Marie Dubois"),
    ("German", "Europe", "Guten Tag, ich heiße Hans Müller und wohne in Berlin.", "Hans Müller"),
    ("Spanish", "Europe", "Hola, me llamo Carlos García y vivo en Madrid.", "Carlos García"),
    ("Portuguese", "Europe", "Olá, meu nome é João Silva e moro em São Paulo.", "João Silva"),
    ("Italian", "Europe", "Ciao, mi chiamo Luca Rossi e abito a Roma.", "Luca Rossi"),
    ("Russian", "Europe", "Привет, меня зовут Владимир Путин, я из Москвы.", "Владимир"),
    ("Dutch", "Europe", "Hallo, ik heet Jan de Vries en ik woon in Amsterdam.", "Jan de Vries"),
    ("Polish", "Europe", "Cześć, nazywam się Jan Kowalski i mieszkam w Warszawie.", "Kowalski"),
    # --- Americas ---
    ("English (US)", "Americas", "Elon Musk founded SpaceX in California.", "Elon Musk"),
    ("Portuguese (BR)", "Americas", "A empresa Petrobras está sediada no Rio de Janeiro.", "Rio"),
    ("Spanish (MX)", "Americas", "Los immigrantes llegaron a Ciudad de México.", "México"),
]


def flatten(nodes: list) -> list:
    """Recursively collect every node (inner entities included)."""
    out: list = []
    for n in nodes:
        out.append(n)
        out.extend(flatten(list(n.children)))
    return out


def main() -> None:
    ner = JevNER(max_workers=4)
    failures: list[str] = []
    print("=" * 70)
    for lang, continent, text, expected in LANGUAGE_CASES:
        forest = ner(text)
        all_nodes = flatten(forest)
        hit_texts = {n.entity.text for n in all_nodes}
        hit = any(expected in t or t in expected for t in hit_texts)
        status = "PASS" if hit else "FAIL"
        if not hit:
            failures.append(lang)
        print(f"[{status}] {lang} ({continent}) expected: {expected}")
        print("  " + "\n  ".join(str(n) for n in forest).replace("\n", "\n  "))
    ner.close()
    print("=" * 70)
    print(f"passed {len(LANGUAGE_CASES) - len(failures)}/{len(LANGUAGE_CASES)} languages")
    if failures:
        print("failed:", ", ".join(failures))
        raise SystemExit(1)
    print("All languages passed.")


if __name__ == "__main__":
    main()
