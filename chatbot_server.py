from flask import Flask, request, jsonify
from flask_cors import CORS  # Importe a biblioteca CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Adicione esta linha para permitir a comunicação

# --- Base de Conhecimento com imagens ---
knowledge_base = [
    {
        "text": """
                    🎓 Acesso ao Computador – Alunos e Professores Senac Camaquã
                    Se é sua primeira vez acessando o PC, não se preocupe! É rapidinho e fácil. Siga os passos abaixo com confiança:
                    👤 Entrar como Novo Usuário
                    Na tela inicial, clique em "Novo Usuário".
                    ✉️ Preenchendo os Campos de Login
                    E-mail: Digite seu CPF seguido de @senacrs.edu.br.
                    👉 Exemplo: 12345678910@senacrs.edu.br
                    Senha: Digite sua data de nascimento no formato DDMMAAAA seguida de #Educ.
                    👉 Exemplo: 30032020#Educ
                    🔐 Importante: Essa senha padrão só funciona se você nunca alterou ou não pediu um reset. Se tiver qualquer dúvida ou problema, é só procurar o Rodrigo na Secretaria — ele está sempre pronto pra ajudar!
                    🧑‍💼 Visitantes
                    Se você é visitante, use este login especial:
                    Login: camaqua01@senacrs.edu.br
                    Senha: Escola.Senac.01
                    ✨ Pronto! Agora é só aproveitar o computador e fazer bom uso dos recursos disponíveis. Seja bem-vindo e bom trabalho ou bons estudos!
                    """,
                            "image_url": "" # Se tiver uma imagem, coloque a URL aqui
    }
]

# Apenas os textos são usados para a busca semântica
texts_for_search = [item['text'] for item in knowledge_base]

print("Carregando modelo SentenceTransformer. Isso pode levar um tempo na primeira vez...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo carregado. Gerando embeddings da base de conhecimento...")
knowledge_embeddings = model.encode(texts_for_search)
print("Embeddings gerados.")


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"text": "Por favor, digite uma pergunta."})

    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, knowledge_embeddings)
    most_similar_index = np.argmax(similarities)
    
    if similarities[0][most_similar_index] < 0.5:
        return jsonify({"text": "Desculpe, não encontrei uma resposta para sua pergunta sobre a Empresa Camaquã. Por favor, tente perguntar de outra forma.", "image_url": ""})
    
    best_match_item = knowledge_base[most_similar_index]
    
    return jsonify({
        "text": best_match_item['text'],
        "image_url": best_match_item['image_url']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)