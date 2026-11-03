"""jevner 语言覆盖扫描：亚洲/欧洲/美洲全语种端到端验证。

对 23 个语种（亚洲 12 / 欧洲 8 / 美洲 3）逐一执行真实 API 的
完整识别管线，校验每个语种都能识别出至少一个人名/地名类实体，
并打印各语种的完整结果供人工审阅。需要环境变量
TYPESAFE_API_KEY / TYPESAFE_BASE_URL，产生真实 API 计费。
"""

from __future__ import annotations

from jevner import JevNER

# (语种, 大洲, 测试文本, 必须命中的实体文本子串)
LANGUAGE_CASES: list[tuple[str, str, str, str]] = [
    # --- 亚洲 ---
    ("汉语", "亚洲", "你好，我的名字是吴子豪，来自福建福州。", "吴子豪"),
    ("日语", "亚洲", "田中さんは東京大学の学生です。", "田中"),
    ("韩语", "亚洲", "김철수 씨는 서울에서 삽니다.", "서울"),
    ("泰语", "亚洲", "สวัสดีครับ ผมชื่อสมชาย มาจากกรุงเทพมหานคร", "สมชาย"),
    ("越南语", "亚洲", "Nguyễn Văn An làm việc tại Hà Nội.", "Hà Nội"),
    ("印尼语", "亚洲", "Presiden Joko Widodo berkunjung ke Jakarta.", "Jakarta"),
    ("马来语", "亚洲", "Hai, nama saya Ahmad, saya berasal dari Kuala Lumpur.", "Kuala Lumpur"),
    ("菲律宾语", "亚洲", "Si Maria ay nagtatrabaho sa Maynila.", "Maria"),
    ("印地语", "亚洲", "राम शर्मा दिल्ली में रहता है।", "राम"),
    ("阿拉伯语", "亚洲", "مرحبا، اسمي محمد وأعمل في شركة أرامكو في الرياض.", "محمد"),
    ("希伯来语", "亚洲", "שלום, שמי דוד ואני גר בירושלים.", "דוד"),
    ("土耳其语", "亚洲", "Ahmet İstanbul'da bir şirkette çalışıyor.", "Ahmet"),
    # --- 欧洲 ---
    ("英语", "欧洲", "Hi, i'm Vortez Wohl, working at Google in London.", "Vortez Wohl"),
    ("法语", "欧洲", "Bonjour, je m'appelle Marie Dubois et j'habite à Paris.", "Marie Dubois"),
    ("德语", "欧洲", "Guten Tag, ich heiße Hans Müller und wohne in Berlin.", "Hans Müller"),
    ("西班牙语", "欧洲", "Hola, me llamo Carlos García y vivo en Madrid.", "Carlos García"),
    ("葡萄牙语", "欧洲", "Olá, meu nome é João Silva e moro em São Paulo.", "João Silva"),
    ("意大利语", "欧洲", "Ciao, mi chiamo Luca Rossi e abito a Roma.", "Luca Rossi"),
    ("俄语", "欧洲", "Привет, меня зовут Владимир Путин, я из Москвы.", "Владимир"),
    ("荷兰语", "欧洲", "Hallo, ik heet Jan de Vries en ik woon in Amsterdam.", "Jan de Vries"),
    ("波兰语", "欧洲", "Cześć, nazywam się Jan Kowalski i mieszkam w Warszawie.", "Kowalski"),
    # --- 美洲 ---
    ("英语(美国)", "美洲", "Elon Musk founded SpaceX in California.", "Elon Musk"),
    ("葡萄牙语(巴西)", "美洲", "A empresa Petrobras está sediada no Rio de Janeiro.", "Rio"),
    ("西班牙语(墨西哥)", "美洲", "Los immigrantes llegaron a Ciudad de México.", "México"),
]


def flatten(nodes: list) -> list:
    """递归收集森林全部节点（含内层实体）。"""
    out: list = []
    for n in nodes:
        out.append(n)
        out.extend(flatten(list(n.children)))
    return out


def main() -> None:
    ner = JevNER()
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
        print(f"[{status}] {lang} ({continent}) 期望: {expected}")
        print("  " + "\n  ".join(str(n) for n in forest).replace("\n", "\n  "))
    ner.close()
    print("=" * 70)
    print(f"通过 {len(LANGUAGE_CASES) - len(failures)}/{len(LANGUAGE_CASES)} 个语种")
    if failures:
        print("失败语种:", ", ".join(failures))
        raise SystemExit(1)
    print("全部语种通过。")


if __name__ == "__main__":
    main()
