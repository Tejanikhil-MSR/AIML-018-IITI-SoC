class QueryClassifier:
    """
    A placeholder class for classifying user queries.
    In a real application, this would involve a trained model.
    """
    def __init__(self):
        # In a real scenario, load your classification model here
        self.available_labels = [
            "Admissions", "Academics", "Student Life", "Research", "Events",
            "General Info", "Retrieval", "Conversational/Greeting"
        ]

    def classify_query(self, query: str) -> list[str]:
        """
        Classifies the input query and returns a list of probable labels.
        This is a simple rule-based example.
        """
        query_lower = query.lower()
        
        probable_labels = []

        # --- New Conversational/Greeting Classification ---
        conversational_keywords = [
            "hi", "hello", "hey", "how are you", "how are u", "what's up",
            "whats up", "good morning", "good afternoon", "good evening",
            "can you help me", "how can you help", "help me", "talk to me",
            "chat with me", "are you there"
        ]
        
        # Check if the query is primarily a greeting or conversational opener
        # We'll be a bit more strict here to avoid false positives for actual queries
        is_conversational = False
        for keyword in conversational_keywords:
            # Using 'in' might be too broad, consider '== keyword' for exact match
            # or checking if the query is very short and matches a greeting.
            # For robustness, we check if the entire (or main part) of the query matches a greeting.
            if keyword in query_lower:
                # Add more sophisticated logic if needed, e.g., to ensure it's not part of a larger, different query
                if len(query_lower.split()) <= 5 or query_lower.strip() == keyword: # Simple length check for greetings
                    probable_labels.append("Conversational/Greeting")
                    is_conversational = True
                    break

        # If it's a conversational query, it might not need other topic classifications.
        # You might decide to return only "Conversational/Greeting" for these.
        # For now, we'll let other classifications also apply if their keywords are present.
        # However, for pure greetings, we might want to prioritize this label.
        if is_conversational and len(probable_labels) == 1: # If only conversational detected so far
            return ["Conversational/Greeting"] # Prioritize returning just this for pure greetings

        # --- Existing Topic-Based Classification ---
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

        # --- Existing Retrieval Classification ---
        # Keywords for Retrieval (seeking existing information)
        retrieval_keywords = [
            "what is", "how to", "tell me about", "information on", "find",
            "get me", "show me", "where is", "when is", "who is", "list of",
            "details about", "query for", "look up"
        ]
        for keyword in retrieval_keywords:
            if keyword in query_lower:
                if "Conversational/Greeting" not in probable_labels or len(query_lower.split()) > len(keyword.split()) + 2: # Avoid conflict with pure greetings
                    probable_labels.append("Retrieval")
                    break

        # If no specific labels are matched, or if it's a very general query
        # This condition now accounts for if a conversational label was already added.
        if not any(label in probable_labels for label in ["Admissions", "Academics", "Student Life", "Research", "Events", "Retrieval"]):
            if "General Info" not in probable_labels: # Avoid duplicates
                probable_labels.append("General Info")

        # Deduplicate and return
        return list(set(probable_labels))

# Instantiate the classifier
query_classifier = QueryClassifier()
