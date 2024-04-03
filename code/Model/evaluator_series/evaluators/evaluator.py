import re
import string
class Evaluator:
    def __init__(self, choices, model_name, k=-1):
        self.choices = choices
        self.model_name = model_name
        self.k = k
        self.puncs = list(string.punctuation)

    def format_example(self, line, include_answer=True):
        pass

    def generate_few_shot_prompt(self, subject, dev_df):
        pass
    
    def eval_MultiLanguage(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None):
        pass

    def normalize_answer(self,s):

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude=set(self.puncs)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self,pred, target):
        return self.normalize_answer(pred)==self.normalize_answer(target)
