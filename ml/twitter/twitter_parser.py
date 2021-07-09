"""Parses / Prepares Single Twitter Message."""
import spacy


class TwitterParser:
    def __init__(self):
        """Construct Twitter Parser."""
        self.nlp = spacy.load("en_core_web_sm")

    def normalize(self, msg):

        # Strip hash tags
        msg = msg.replace("#", "")

        # Convert to lowercase
        msg = msg.lower()

        doc = self.nlp(msg)
        #self.debug_doc(doc)

        final_tokens = []
        for token in doc:
            if not token.is_stop:
                final_tokens.append(token.lemma_)
        return final_tokens

    def debug_doc(self, doc):
        for token in doc:
            print(
                token.text,
                token.lemma_,
                token.pos_,
                token.tag_,
                token.dep_,
                token.shape_,
                token.is_alpha,
                token.is_stop,
            )