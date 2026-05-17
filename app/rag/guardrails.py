SEMO_TOPICS = [
    "admission", "enrollment", "tuition", "financial aid", "scholarship",
    "course", "curriculum", "semester", "faculty", "professor", "campus",
    "library", "housing", "dining", "transcript", "degree", "graduation",
    "department", "undergraduate", "graduate", "research", "exam", "registration",
    "academic calendar", "student services", "international student",
    "semo", "southeast missouri", "redhawks", "cape girardeau",
    "college of business", "college of education", "college of science",
    "college of liberal arts", "nursing", "polytechnic", "honors",
    "financial aid", "fafsa", "bursar", "registrar", "advising",
    "dorm", "residence hall", "meal plan", "student id", "parking",
    "commencement", "athletics", "greek life", "student organization",
]

OFF_TOPIC_RESPONSE = (
    "I'm the SEMO Assistant and can only answer questions about "
    "Southeast Missouri State University — admissions, courses, campus life, "
    "financial aid, student services, and university policies. "
    "Please ask a SEMO-related question."
)

def is_university_topic(query: str) -> bool:
    q = query.lower()
    return any(topic in q for topic in SEMO_TOPICS)

def get_domain_system_prompt() -> str:
    return """You are the SEMO Assistant — a friendly Q&A assistant for Southeast Missouri State University (semo.edu).

STRICT RULES:
1. ONLY answer questions about Southeast Missouri State University: academics, admissions, student life, courses, faculty, campus facilities, financial aid, university policies, and related SEMO topics.
2. If a question is not related to SEMO, respond ONLY with: "I can only assist with SEMO-related questions."
3. Base answers SOLELY on the provided context passages.
4. Do NOT include inline source citations like [Source: ...] in your answer — sources are shown separately.
5. If context is insufficient, say: "I don't have enough information about that SEMO topic."

Format your answer using clear markdown: use headers, bullet points, and bold text to make it easy to read."""
