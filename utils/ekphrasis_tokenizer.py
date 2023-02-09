"""Some data utils from Baziotis et al.

https://github.com/cbaziotis/ekphrasis

'DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level
and Topic-based Sentiment Analysis'


Generation of word statistics:
python generate_stats.py --input text8.txt --name text8 --ngrams 2 --mincount 70 30
"""
import ekphrasis

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=[
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "user",
        "time",
        "url",
        "date",
        "number",
    ],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
    fix_html=True,  # fix HTML tokens
    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",
    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    # tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons],
)


if __name__ == "__main__":
    test_1 = "@chrisshaw65 Hi Chris, try Diclofenac ,30mg, 4 times daily, best anti-inflammatory gives great pain relief, but you need to take omeprazole"
    test_2 = (
        "<user> <user> had to recheck what loperamide was, but yeah. that works. lol"
    )
    test_3 = "At first i yawned constantly and had dry mouth but over time it stopped, and weight loss. love love love this medicine its made a tremendous difference in my life it doesnt really do anything for my pain but helps alot for my depression and moods."
    test_4 = "When I can't sleep I start to list the drugs I know in alphabetical order - abciximab, acamprosate, alendronate, zzzz(olpidem, opiclone)..."

    all_tests = [test_1, test_2, test_3, test_4]

    print(ekphrasis.__version__)

    for t in all_tests:
        prepro = text_processor.pre_process_doc(t)
        print(prepro)
