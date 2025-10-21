
from nltk.corpus import wordnet as wn
from utils import replace_code


# pip install regex
import regex

_EMOJI_ONLY = regex.compile(
    r"[\p{Emoji_Presentation}"
    r"\U0001F300-\U0001F9FF"   # 大部分彩色符号
    r"\U0001FA70-\U0001FAFF"   # 扩展 A
    r"\U00002600-\U000027BF"   # 杂项符号
    r"]+",
    regex.V1,
)

def strip_emoji(text: str) -> str:
    return _EMOJI_ONLY.sub("", text)



a = """
************ Origin Code Start ************
['# Proposed action: [tool 4]\n# This is a placeholder for the actual clue and number. Let\'s assume the clue is "touch" and the number is 3.\n\n# List of Codenames Words\ncodenames_words = {\n    "glove": "N", \n    "knife": "R", \n    "name": "N", \n    "silk": "B", \n    "branch": "N", \n    "copy": "R", \n    "office": "B", \n    "hate": "A", \n    "weather": "N", \n    "limit": "N", \n    "day": "B", \n    "flight": "R", \n    "touch": "R", \n    "sneeze": "N", \n    "profit": "B", \n    "end": "B", \n    "request": "B", \n    "crack": "R", \n    "range": "N", \n    "leaf": "R", \n    "boy": "R", \n    "owner": "R", \n    "horse": "B", \n    "cat": "R", \n    "off": "B"\n}\n\n# Proposed clue and number\nproposed_clue = "touch"\nproposed_number = 3\n\n# Check if the clue is in the Codenames Words list\nif proposed_clue in codenames_words:\n    print(f"Error: The clue \'{proposed_clue}\' is present in the Codenames Words list. This is not allowed.")\nelse:\n    print(f"Valid clue: \'{proposed_clue}\' is not present in the Codenames Words list.")\n\n# Print the proposed action\nprint(f"Proposed action: [{proposed_clue} {proposed_number}]")']
************ Origin Code End ************
"""

if __name__ == "__main__":
    print(a)
