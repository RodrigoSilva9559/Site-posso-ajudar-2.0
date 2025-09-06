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
    },
    {
        "text": """
                    🧠 Como ativar o pacote Office

                    É super fácil! Siga os passos abaixo com confiança:

                    1.  Abra o Word, Excel ou PowerPoint.
                    2.  Vai aparecer uma janelinha automática — clique em "Entrar ou Criar uma conta".
                    3.  Digite seu e-mail: SeuCPF@senacrs.edu.br
                    4.  Insira sua senha (a mesma que você usa para fazer login no computador).
                    5.  Se o sistema pedir para trocar a senha, siga os passos e clique em Enviar.
                    6.  Aceite o contrato de licença e feche a tela final.

                    ✅ Pronto! Office ativado e liberado para uso!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    🧾 Solicitar serviços da Secretaria pelo Portal do Aluno

                    Você pode solicitar serviços de um jeito bem prático!

                    1.  Acesse o site Senac-RS.
                    2.  Clique no ícone de usuário (canto superior direito) e selecione Portal do Aluno.
                    3.  Faça login com seu CPF ou matrícula e data de nascimento (DDMMAAAA).
                    4.  Se for seu primeiro acesso, o sistema pedirá para trocar a senha.
                    5.  Vá em Ambiente do Estudante > Autoatendimento > Solicitar Serviço.
                    6.  Escolha o serviço desejado (ex: Justificativa de Faltas), selecione o curso e descreva o motivo.
                    7.  Anexe o documento (foto ou arquivo) e clique em Enviar. 🎯 Você verá uma mensagem confirmando que deu tudo certo!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📚 Acessar a Biblioteca Online

                    Nossa biblioteca virtual está cheia de conteúdo incrível!

                    1.  Entre no site do Senac-RS.
                    2.  Vá até Serviços > Bibliotecas.
                    3.  Clique no logo Minha Biblioteca.
                    4.  Faça login com seu usuário e senha (solicite na Secretaria se ainda não tiver).

                    📖 Aproveite mais de 11 mil livros digitais!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    🔐 Trocar sua senha no computador

                    Para manter sua conta segura, siga estes passos para trocar sua senha:

                    1.  Pressione Ctrl + Alt + Delete.
                    2.  Clique em Alterar uma senha.
                    3.  Digite a senha atual, a nova e confirme.
                    4.  Clique em Enviar.

                    💡 Dica: Crie uma senha forte com letras maiúsculas, minúsculas, números e símbolos!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📖 Empréstimo de Livros na Biblioteca

                    Adora um bom livro? A gente facilita o empréstimo!

                    1.  Escolha seu livro na estante da área de convivência.
                    2.  Leve até a Secretaria para registrar.

                    O prazo de empréstimo é de 7 dias corridos, com a possibilidade de renovação por mais 7 dias.

                    ⚠️ Multa de R$ 1,00 por dia de atraso.
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    🎓 Conheça nossos cursos

                    Temos uma variedade de cursos para ajudar você a crescer!

                    * Cursos Livres (FIC)
                    * Cursos Técnicos
                    * Cursos EAD – FIC, Técnico, Graduação, Pós e Extensão

                    📍 Selecione Senac Camaquã para ver as turmas abertas!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📲 Fale com nossas consultoras

                    Precisa de ajuda para encontrar o curso ideal? Nossas consultoras estão prontas para te ajudar!

                    * Laurielle: WhatsApp
                    * Thais: WhatsApp
                    * Tailine: WhatsApp
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📡 Senha do Wi-Fi

                    A senha da nossa rede de visitantes é super fácil de lembrar!

                    Rede: Senac Visitantes
                    Senha: trijuntos
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📜 Solicitar certificado, diploma ou atestado escolar

                    Você pode fazer essa solicitação pelo Portal do Aluno!

                    1.  Acesse o Portal do Aluno.
                    2.  Vá em Ambiente do Estudante > Página Principal > Atestado de Matrícula.

                    Para atestados personalizados, abra um protocolo em Autoatendimento > Solicitar Serviço.

                    ⏳ Prazo: até 6 dias úteis.
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    💳 Pagamento de cursos

                    Se precisar de ajuda para pagar seu curso, temos várias opções!

                    1.  Acesse o Portal do Aluno > Página Principal > Financeiro.
                    2.  Escolha o período, selecione seu nome e veja as opções: boleto, Pix ou cartão.

                    Você também pode pagar direto na escola com Tainã ou Tatiane ou pelo WhatsApp do Financeiro.
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📈 Boletim Escolar

                    Quer ver suas notas? É só acessar o Portal do Aluno e ir em Boletim. Depois, selecione o curso para ver seu desempenho!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    🎓 Cursos Gratuitos – PSG

                    Temos cursos gratuitos para você!

                    Acesse: Consulta de Vagas PSG
                    Pesquise sua cidade, veja os cursos e inscreva-se!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    💼 Trabalhe no Senac Camaquã

                    Quer fazer parte do nosso time? Fique de olho nas nossas vagas!

                    Acesse: Trabalhe Conosco
                    Cadastre seu currículo e acompanhe as vagas!
                    """,
                            "image_url": ""
    },
    {
        "text": """
                    📅 Justificativa de Faltas

                    Para justificar uma falta, você precisa de um documento válido, como: atestados médicos, convocação judicial, licença maternidade, etc.

                    Envie pelo Portal do Aluno em até 2 dias úteis com foto ou escaneamento legível.
                    """,
                            "image_url": ""
    }
]

# Apenas os textos são usados para a busca semântica
texts_for_search = [item['text'] for item in knowledge_base]

print("Carregando modelo SentenceTransformer. Isso pode levar um tempo na primeira vez...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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