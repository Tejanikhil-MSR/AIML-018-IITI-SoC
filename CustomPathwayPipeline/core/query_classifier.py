class QueryClassifier:
    """
    A generic class for classifying user queries based on a provided keyword-to-label map.
    This is a simple rule-based example, extensible by configuring the keyword map.                                       
    """
    def __init__(self, intent_keywords_map: dict = None, default_label: str = "General Info"):
        """
        Initializes the QueryClassifier with a map of intent labels to keywords.

        Args:
            intent_keywords_map (dict, optional): A dictionary where keys are intent labels (str)
                                                    and values are lists of keywords (list[str])
                                                    associated with that intent.
                                                    If None, a default map based on INTENTS from config
                                                    will be used.
            default_label (str, optional): The label to assign if no specific intent is found.
                                            Defaults to "General Info".
        """
        if intent_keywords_map is None:
            # Provide a sensible default mapping if none is given,
            # using the INTENTS from your config.py
            self.intent_keywords_map = {
                "Admissions": ["admission", "apply", "entrance", "eligibility"],
                "Academics": ["course", "program", "syllabus", "department", "academics"],
                "Student Life": ["hostel", "mess", "club", "sports", "student life"],
                "Research": ["research", "project", "publication", "phd"],
                "Events": ["event", "workshop", "conference", "seminar"],
            }
        else:
            self.intent_keywords_map = intent_keywords_map

        self.available_labels = list(self.intent_keywords_map.keys()) # Dynamically get available labels
        self.default_label = default_label


    def classify_query(self, query: str) -> list[str]:
        """
        Classifies the input query and returns a list of probable labels
        based on the initialized keyword map.
        """
        query_lower = query.lower()
        probable_labels = []

        for label, keywords in self.intent_keywords_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    probable_labels.append(label)
                    break # Move to the next label once a match is found for the current label

        # Add the default label if no specific labels were found
        if not probable_labels and self.default_label:
            probable_labels.append(self.default_label)

        return list(set(probable_labels)) # Return unique labels

# Example of how to use it with default settings (similar to original usage)
# query_classifier = QueryClassifier()

# Example of how to use it with custom intent keywords
# custom_intent_map = {
#     "Campus Tours": ["tour", "visit", "campus"],
#     "Faculty Info": ["professor", "faculty", "department head"],
#     "General Inquiries": ["general", "info", "question"]
# }
# custom_query_classifier = QueryClassifier(
#     intent_keywords_map=custom_intent_map,
#     default_label="Other"
# )

# Test the custom classifier
# print(custom_query_classifier.classify_query("I want to know about campus tours."))
# print(custom_query_classifier.classify_query("Who is the head of the computer science department?"))
# print(custom_query_classifier.classify_query("What is the weather like?"))