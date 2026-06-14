from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "تعلم بايثون يساعد على تحليل البيانات وبناء التطبيقات",
    "الذكاء الاصطناعي يستخدم في معالجة اللغة الطبيعية",
    "محركات البحث تسترجع المستندات المرتبطة باستعلام المستخدم",
    "كرة القدم من أكثر الرياضات شعبية حول العالم",
    "تعلم الآلة أحد فروع الذكاء الاصطناعي"
]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

def search(query, top_k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for index in ranked_indices:
        if scores[index] > 0:
            results.append({
                "document": documents[index],
                "score": round(float(scores[index]), 3)
            })

    return results

results = search("ما هي محركات البحث؟")

for result in results:
    print(result["score"], "-", result["document"])