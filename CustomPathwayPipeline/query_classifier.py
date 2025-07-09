# query_classifier.py

class QueryClassifier:
    """
    A placeholder class for classifying user queries.
    In a real application, this would involve a trained model.
    """
    def __init__(self):
        # In a real scenario, load your classification model here
        self.available_labels = ["Admissions", "Academics", "Student Life", "Research", "Events", "General Info"]

    def classify_query(self, query: str) -> list[str]:
        """
        Classifies the input query and returns a list of probable labels.
        This is a simple rule-based example.
        """
        query_lower = query.lower()
        
        probable_labels = []

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
        
        # If no specific labels are matched, or if it's a very general query
        if not probable_labels or "general" in query_lower or "info" in query_lower:
            if "General Info" not in probable_labels: # Avoid duplicates
                probable_labels.append("General Info")
        
        
        # return list(set(probable_labels))
        return ["admissions", "course", "hostel", "research"]
        
# Instantiate the classifier
query_classifier = QueryClassifier()