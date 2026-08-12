# ============================================================
# DATA AND CONFIGURATION
# ============================================================

# ============================================================
# DATASET RANGES
# Based on the 2200-row dataset
# ============================================================

DATASET_RANGES = {
    "N": (0.0, 140.0),
    "P": (5.0, 145.0),
    "K": (5.0, 205.0),
    "temperature": (8.825675, 43.675493),
    "humidity": (14.258040, 99.981876),
    "ph": (3.504752, 9.935091),
    "rainfall": (20.211267, 298.560117)
}


# ============================================================
# CROP INFORMATION
# ============================================================

CROP_INFO = {

    "rice": {
        "emoji": "🌾",
        "description":
            "A water-demanding cereal crop commonly associated "
            "with relatively humid and high-rainfall conditions.",
        "advice":
            "Consider local water availability and seasonal rainfall "
            "before cultivation."
    },

    "maize": {
        "emoji": "🌽",
        "description":
            "A major cereal crop whose suitability depends on the "
            "combined soil and environmental conditions.",
        "advice":
            "Consider local climate, soil fertility and water availability."
    },

    "chickpea": {
        "emoji": "🫘",
        "description":
            "A pulse crop whose recommendation depends on the "
            "combination of nutrient and environmental conditions.",
        "advice":
            "Check local season and soil conditions before cultivation."
    },

    "kidneybeans": {
        "emoji": "🫘",
        "description":
            "A pulse crop selected when the measured conditions "
            "match patterns learned from the training data.",
        "advice":
            "Verify local growing-season suitability."
    },

    "pigeonpeas": {
        "emoji": "🫘",
        "description":
            "A pulse crop whose recommendation is based on the "
            "combined soil and environmental measurements.",
        "advice":
            "Consider local rainfall and seasonal conditions."
    },

    "mothbeans": {
        "emoji": "🌱",
        "description":
            "A pulse crop selected based on patterns learned "
            "from soil and environmental measurements.",
        "advice":
            "Check local water availability before planting."
    },

    "mungbean": {
        "emoji": "🌱",
        "description":
            "A pulse crop recommended when the input conditions "
            "match learned patterns for mungbean.",
        "advice":
            "Consider local temperature and rainfall conditions."
    },

    "blackgram": {
        "emoji": "🌱",
        "description":
            "A pulse crop whose prediction depends on all seven "
            "soil and environmental features.",
        "advice":
            "Use the recommendation together with local agricultural advice."
    },

    "lentil": {
        "emoji": "🌱",
        "description":
            "A pulse crop recommended according to patterns "
            "learned from the training dataset.",
        "advice":
            "Consider the local season before cultivation."
    },

    "pomegranate": {
        "emoji": "🍎",
        "description":
            "A fruit crop selected according to the combined "
            "soil and environmental conditions.",
        "advice":
            "Consider long-term climate and irrigation availability."
    },

    "banana": {
        "emoji": "🍌",
        "description":
            "A fruit crop whose recommendation depends on "
            "the combination of the seven input features.",
        "advice":
            "Consider water availability and local climate."
    },

    "mango": {
        "emoji": "🥭",
        "description":
            "A fruit crop selected when the measured conditions "
            "match patterns learned for mango.",
        "advice":
            "Consider long-term climate suitability."
    },

    "grapes": {
        "emoji": "🍇",
        "description":
            "A fruit crop recommended according to patterns "
            "learned from the soil and environmental data.",
        "advice":
            "Consider local climate and irrigation conditions."
    },

    "watermelon": {
        "emoji": "🍉",
        "description":
            "A crop whose prediction depends on the combined "
            "temperature, humidity, soil and rainfall conditions.",
        "advice":
            "Consider water availability during the growing period."
    },

    "muskmelon": {
        "emoji": "🍈",
        "description":
            "A crop selected according to patterns learned "
            "from the seven model inputs.",
        "advice":
            "Consider temperature and water availability."
    },

    "apple": {
        "emoji": "🍎",
        "description":
            "A fruit crop selected when the environmental and "
            "soil measurements resemble learned patterns.",
        "advice":
            "Consider local climate suitability."
    },

    "orange": {
        "emoji": "🍊",
        "description":
            "A fruit crop whose prediction uses the combined "
            "soil and environmental measurements.",
        "advice":
            "Consider local climate and irrigation conditions."
    },

    "papaya": {
        "emoji": "🥭",
        "description":
            "A fruit crop that can be recommended under suitable "
            "temperature, humidity, soil and rainfall conditions.",
        "advice":
            "Consider local water availability and climate."
    },

    "coconut": {
        "emoji": "🥥",
        "description":
            "A crop whose recommendation can be associated with "
            "humid and rainfall-related environmental conditions.",
        "advice":
            "Consider long-term rainfall and water availability."
    },

    "cotton": {
        "emoji": "🌿",
        "description":
            "A commercial crop whose prediction depends on the "
            "combined soil and environmental measurements.",
        "advice":
            "Consider local climate and market conditions."
    },

    "jute": {
        "emoji": "🌿",
        "description":
            "A crop whose recommendation depends on moisture, "
            "rainfall, temperature and soil characteristics.",
        "advice":
            "Consider local rainfall and soil conditions."
    },

    "coffee": {
        "emoji": "☕",
        "description":
            "A crop selected when the measured conditions resemble "
            "patterns learned for coffee.",
        "advice":
            "Consider local climate, shade and water availability."
    }
}