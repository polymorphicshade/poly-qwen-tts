"""Languages and the built-in voice library.

The library has two kinds of built-in voice:

* ``native`` — the 9 speakers baked into the Qwen3-TTS CustomVoice checkpoint. They accept style
  instructions directly.
* ``designed`` — 41 timbres defined by a natural-language description. The first time one is used,
  a reference clip is rendered with the VoiceDesign model and cached, and later plain requests clone
  that clip with the Base model so the timbre stays the same across generations. A request with a
  style instruction goes back to VoiceDesign with the description plus the instruction.

Voices cloned by the user (``cloned``) are stored on disk by :mod:`poly_tts.library`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

LANGUAGES: list[str] = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

_LANGUAGE_ALIASES = {
    "auto": "Auto",
    "zh": "Chinese", "cn": "Chinese", "mandarin": "Chinese", "chinese": "Chinese",
    "en": "English", "english": "English",
    "ja": "Japanese", "jp": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "kr": "Korean", "korean": "Korean",
    "de": "German", "german": "German",
    "fr": "French", "french": "French",
    "ru": "Russian", "russian": "Russian",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "es": "Spanish", "spanish": "Spanish",
    "it": "Italian", "italian": "Italian",
}


def normalize_language(value: str | None) -> str:
    """Map ``"en"``, ``"english"``, ``"English"`` … to the canonical model language name."""
    if value is None or value == "":
        return "Auto"
    try:
        return _LANGUAGE_ALIASES[value.strip().lower()]
    except KeyError:
        raise ValueError(f"Unsupported language {value!r}. Supported: {', '.join(LANGUAGES)}") from None


# Sentences used to render reference clips for designed presets (roughly 6-10 seconds each).
ANCHOR_TEXT: dict[str, str] = {
    "Chinese": "你好，很高兴认识你。今天天气不错，我们一起出去走走，顺便聊聊最近发生的有趣事情吧。",
    "English": "Hello, it's nice to meet you. The weather is lovely today, so let's take a walk and talk about everything that's happened lately.",
    "Japanese": "こんにちは、はじめまして。今日はいい天気ですね。少し散歩しながら、最近あった面白い話をしましょう。",
    "Korean": "안녕하세요, 만나서 반가워요. 오늘 날씨가 정말 좋네요. 같이 산책하면서 요즘 있었던 재미있는 이야기를 나눠요.",
    "German": "Hallo, schön dich kennenzulernen. Das Wetter ist heute herrlich, also lass uns spazieren gehen und über die letzten Wochen reden.",
    "French": "Bonjour, ravi de vous rencontrer. Il fait très beau aujourd'hui, alors allons nous promener et parler de tout ce qui s'est passé récemment.",
    "Russian": "Здравствуйте, рад познакомиться. Сегодня прекрасная погода, давайте прогуляемся и поговорим обо всём, что случилось за последнее время.",
    "Portuguese": "Olá, é um prazer conhecer você. O tempo está ótimo hoje, então vamos dar uma volta e conversar sobre tudo o que aconteceu ultimamente.",
    "Spanish": "Hola, encantado de conocerte. Hace un día precioso, así que vamos a dar un paseo y a hablar de todo lo que ha pasado últimamente.",
    "Italian": "Ciao, piacere di conoscerti. Oggi è una bellissima giornata, quindi facciamo una passeggiata e parliamo di tutto quello che è successo.",
}


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    kind: str  # native | designed | cloned
    gender: str  # female | male | neutral
    age: str  # child | young | adult | middle-aged | senior
    language: str  # native / recommended language
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    supports_style: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


NATIVE_VOICES: list[Voice] = [
    Voice("vivian", "Vivian", "native", "female", "young", "Chinese", "Bright, slightly edgy young female voice.", ("bright", "lively")),
    Voice("serena", "Serena", "native", "female", "young", "Chinese", "Warm, gentle young female voice.", ("warm", "gentle")),
    Voice("uncle_fu", "Uncle Fu", "native", "male", "senior", "Chinese", "Seasoned male voice with a low, mellow timbre.", ("low", "mellow")),
    Voice("dylan", "Dylan", "native", "male", "young", "Chinese", "Youthful Beijing male voice with a clear, natural timbre.", ("beijing-dialect", "clear")),
    Voice("eric", "Eric", "native", "male", "adult", "Chinese", "Lively Chengdu male voice, slightly husky and bright.", ("sichuan-dialect", "husky")),
    Voice("ryan", "Ryan", "native", "male", "adult", "English", "Dynamic male voice with a strong sense of rhythm.", ("dynamic", "rhythmic")),
    Voice("aiden", "Aiden", "native", "male", "young", "English", "Sunny American male voice with a clear midrange.", ("american", "sunny")),
    Voice("ono_anna", "Ono Anna", "native", "female", "young", "Japanese", "Playful Japanese female voice with a light, nimble timbre.", ("playful", "light")),
    Voice("sohee", "Sohee", "native", "female", "adult", "Korean", "Warm Korean female voice with rich emotion.", ("warm", "emotive")),
]

# Model-side speaker ids for native voices.
NATIVE_SPEAKER_IDS: dict[str, str] = {
    "vivian": "Vivian", "serena": "Serena", "uncle_fu": "Uncle_Fu", "dylan": "Dylan", "eric": "Eric",
    "ryan": "Ryan", "aiden": "Aiden", "ono_anna": "Ono_Anna", "sohee": "Sohee",
}


def _d(id: str, name: str, gender: str, age: str, language: str, description: str, *tags: str) -> Voice:
    return Voice(id, name, "designed", gender, age, language, description, tuple(tags))


DESIGNED_VOICES: list[Voice] = [
    # English
    _d("narrator_deep", "Atlas", "male", "middle-aged", "English", "A deep, resonant middle-aged male voice with a slight rasp, speaking in a calm, authoritative documentary-narrator tone at a measured pace.", "narrator", "deep"),
    _d("audiobook_warm", "Harper", "female", "adult", "English", "A warm, intimate adult female voice with a soft American accent, ideal for reading novels aloud; smooth, expressive and unhurried.", "audiobook", "warm"),
    _d("news_anchor", "Morgan", "male", "adult", "English", "A crisp, polished male news anchor with neutral American pronunciation, steady pace and confident, objective delivery.", "news", "crisp"),
    _d("british_butler", "Reginald", "male", "senior", "English", "An elderly British gentleman with a refined Received Pronunciation accent, dry wit and impeccably formal, unhurried diction.", "british", "formal"),
    _d("cheerful_kid", "Milo", "male", "child", "English", "An energetic eight-year-old boy with a high, bright voice, talking quickly and excitedly with lots of curiosity.", "child", "energetic"),
    _d("little_girl", "Lily", "female", "child", "English", "A sweet six-year-old girl with a small, high-pitched voice, giggly and innocent, speaking slowly and carefully.", "child", "sweet"),
    _d("teen_girl", "Zoe", "female", "young", "English", "A sarcastic sixteen-year-old American girl with a slightly nasal, upbeat voice and casual, fast-paced valley-girl delivery.", "teen", "sarcastic"),
    _d("grandma_story", "Nana Rose", "female", "senior", "English", "A kind grandmother in her eighties with a soft, slightly shaky and breathy voice, telling bedtime stories with gentle warmth.", "elderly", "storyteller"),
    _d("old_wizard", "Aldric", "male", "senior", "English", "An ancient wizard with a gravelly, cracked low voice, speaking slowly and mysteriously with dramatic pauses.", "fantasy", "gravelly"),
    _d("drill_sergeant", "Sergeant Kane", "male", "adult", "English", "A loud, gruff military drill sergeant with a barking, commanding voice and clipped, aggressive delivery.", "military", "loud"),
    _d("radio_dj", "Jax", "male", "young", "English", "A high-energy late-night radio DJ with a smooth, punchy baritone and playful, hype-filled delivery.", "radio", "energetic"),
    _d("asmr_whisper", "Willow", "female", "young", "English", "A soft-spoken young woman with a breathy, close-mic whisper, calm and soothing, speaking very slowly.", "asmr", "whisper"),
    _d("customer_support", "Ava", "female", "adult", "English", "A friendly, patient customer-support agent with a clear, bright mid-pitched voice and reassuring, professional tone.", "assistant", "friendly"),
    _d("noir_detective", "Sam Marlowe", "male", "middle-aged", "English", "A weary 1940s film-noir detective with a low, smoky voice, cynical and laconic, delivering lines with a slow drawl.", "noir", "smoky"),
    _d("southern_belle", "Savannah", "female", "adult", "English", "A charming woman from the American South with a warm, honeyed drawl and a relaxed, melodic rhythm.", "southern", "drawl"),
    _d("aussie_surfer", "Kai", "male", "young", "English", "A laid-back young Australian surfer with a sunny, relaxed voice and a broad Aussie accent.", "australian", "relaxed"),
    _d("scottish_bard", "Fiona", "female", "adult", "English", "A lively Scottish woman with a lilting Highland accent, warm and humorous, with a musical storytelling cadence.", "scottish", "lilting"),
    _d("villain_cold", "Viktor", "male", "adult", "English", "A cold, menacing villain with a quiet, silky low voice, speaking slowly and precisely with barely concealed contempt.", "villain", "menacing"),
    _d("robot_ai", "Unit-7", "neutral", "adult", "English", "A calm, synthetic-sounding AI assistant with an even, slightly monotone mid-pitched voice and precise, clipped articulation.", "robotic", "neutral"),
    _d("sports_commentator", "Rex", "male", "adult", "English", "An explosive sports commentator with a booming, rapid-fire voice that rises to excited shouts at big moments.", "sports", "excited"),
    _d("meditation_guide", "Sage", "neutral", "adult", "English", "A serene, androgynous meditation guide with a soft, low, warm voice and a slow, spacious, breath-like pace.", "calm", "meditation"),
    _d("shy_student", "Theo", "male", "young", "English", "A shy, nervous college student with a soft, slightly hesitant tenor voice who trails off at the ends of sentences.", "shy", "hesitant"),
    _d("ceo_executive", "Victoria", "female", "middle-aged", "English", "A confident female executive in her fifties with a low, controlled alto voice and decisive, persuasive delivery.", "corporate", "confident"),
    # Chinese
    _d("zh_storyteller", "老陈 (Lao Chen)", "male", "senior", "Chinese", "一位年过六旬的说书先生，嗓音浑厚沙哑，抑扬顿挫，语速时快时慢，富有感染力。", "storyteller", "mandarin"),
    _d("zh_anchor", "林晓 (Lin Xiao)", "female", "adult", "Chinese", "一位标准普通话的女新闻主播，声音清亮端庄，吐字清晰，语速平稳，语气客观专业。", "news", "mandarin"),
    _d("zh_youth", "小宇 (Xiao Yu)", "male", "young", "Chinese", "一个阳光开朗的二十岁男大学生，声音清爽明亮，语速偏快，说话带着笑意。", "young", "sunny"),
    _d("zh_cute_girl", "糖糖 (Tangtang)", "female", "young", "Chinese", "一个甜美可爱的少女，音调偏高，声音软糯，说话带点撒娇的语气。", "cute", "sweet"),
    # Japanese
    _d("ja_anime_hero", "Haruto", "male", "young", "Japanese", "熱血アニメの主人公のような、明るく力強い少年の声。情熱的で、語尾に勢いがある。", "anime", "energetic"),
    _d("ja_calm_woman", "Yuki", "female", "adult", "Japanese", "落ち着いた大人の女性の声。柔らかく上品で、ゆっくりと丁寧に話す。", "calm", "elegant"),
    # Korean
    _d("ko_idol", "Minji", "female", "young", "Korean", "밝고 상큼한 20대 여성 아이돌 목소리. 에너지가 넘치고 애교 섞인 말투로 빠르게 말한다.", "idol", "bright"),
    _d("ko_drama_lead", "Joon", "male", "adult", "Korean", "드라마 남자 주인공처럼 낮고 부드러운 30대 남성 목소리. 차분하고 진지하며 감정이 깊다.", "drama", "smooth"),
    # German
    _d("de_professor", "Professor Klaus", "male", "senior", "German", "Ein älterer deutscher Professor mit tiefer, ruhiger Stimme, präziser Aussprache und nachdenklichem, belehrendem Ton.", "academic", "precise"),
    _d("de_bright_host", "Lena", "female", "young", "German", "Eine fröhliche junge Moderatorin mit heller, klarer Stimme und lebhaftem, freundlichem Tempo.", "host", "bright"),
    # French
    _d("fr_parisian", "Camille", "female", "adult", "French", "Une Parisienne élégante à la voix douce et légèrement voilée, au débit détendu et au ton charmeur.", "parisian", "elegant"),
    _d("fr_chef", "Chef Lucien", "male", "middle-aged", "French", "Un chef cuisinier passionné à la voix chaleureuse et sonore, expressif, parlant vite avec beaucoup d'enthousiasme.", "chef", "passionate"),
    # Russian
    _d("ru_bass", "Dmitri", "male", "middle-aged", "Russian", "Глубокий бархатный бас мужчины средних лет, спокойная, весомая и неторопливая манера речи.", "bass", "deep"),
    _d("ru_teacher", "Natalia", "female", "adult", "Russian", "Строгая, но добрая учительница с ясным, звонким голосом и чёткой, размеренной дикцией.", "teacher", "clear"),
    # Portuguese
    _d("pt_carioca", "Thiago", "male", "young", "Portuguese", "Um jovem carioca descontraído, com voz calorosa e sotaque do Rio de Janeiro, falando de forma animada e musical.", "brazilian", "relaxed"),
    # Spanish
    _d("es_telenovela", "Valentina", "female", "adult", "Spanish", "Una actriz de telenovela con voz apasionada y dramática, de timbre cálido y ritmo intenso y emotivo.", "dramatic", "passionate"),
    _d("es_abuelo", "Don Ernesto", "male", "senior", "Spanish", "Un abuelo mexicano bondadoso, de voz grave y cálida, algo ronca, que habla despacio y con cariño.", "elderly", "warm"),
    # Italian
    _d("it_tenor", "Marco", "male", "adult", "Italian", "Un uomo italiano espressivo con una voce da tenore luminosa e calda, che parla con entusiasmo e gesti vocali ampi.", "expressive", "warm"),
]

BUILTIN_VOICES: list[Voice] = NATIVE_VOICES + DESIGNED_VOICES
_BUILTIN_BY_ID = {v.id: v for v in BUILTIN_VOICES}


def get_builtin_voice(voice_id: str) -> Voice | None:
    return _BUILTIN_BY_ID.get(voice_id.lower())
