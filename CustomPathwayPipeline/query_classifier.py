from config import INTENTS

class QueryClassifier:
    """
    A placeholder class for classifying user queries.
    In a real application, this would involve a trained model.
    """
    def __init__(self):
        self.available_labels = INTENTS

    def classify_query(self, query: str) -> list[str]:
        """
        Classifies the input query and returns a list of probable labels.
        This is a simple rule-based example.
        """
        query_lower = query.lower()
        
        probable_labels = []
        
        # is_conversational = False
        # for keyword in DIRECT_RESPONSE_LABELS:
        #     if keyword in query_lower:
        #         if len(query_lower.split()) <= 5 or query_lower.strip() == keyword: 
        #             probable_labels.append("Conversational/Greeting")
        #             is_conversational = True
        #             break

        # if is_conversational and len(probable_labels) == 1:
        #     return ["Conversational/Greeting"] 

        if "admission" in query_lower or "apply" in query_lower or "entrance" in query_lower or "eligibility" in query_lower:
            probable_labels.append("Admissions")
        if "course" in query_lower or "program" in query_lower or "syllabus" in query_lower or "department" in query_lower:
            probable_labels.append("Academics")
        if "hostel" in query_lower or "mess" in query_lower or "club" in query_lower or "sports" in query_lower:
            probable_labels.append("Student Life")
        if "research" in query_lower or "project" in query_lower or "publication" in query_lower or "phd" in query_lower:
            probable_labels.append("Research")
        if "event" in query_lower or "workshop" in query_lower or "conference" in query_lower or "seminar" in query_lower:
            probable_labels.append("Events")

        if not any(label in probable_labels for label in ["Admissions", "Academics", "Student Life", "Research", "Events", "Retrieval"]):
            if "General Info" not in probable_labels:
                probable_labels.append("General Info")

        return list(set(probable_labels))

query_classifier = QueryClassifier()
