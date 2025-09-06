document.addEventListener('DOMContentLoaded', () => {
    // Variáveis do Chatbot
    const assistente = document.querySelector('.assistente-virtual');
    const iconeChat = document.querySelector('.icone-assistente');
    const popupChat = document.querySelector('.popup-assistente');
    const botaoMinimizar = document.getElementById('minimizar-chat');
    const inputUsuario = document.getElementById('input-usuario');
    const botaoEnviar = document.getElementById('enviar-pergunta');
    const conversaContainer = document.getElementById('conversa-container');

    // Variáveis do Pop-up de Feedback
    const popupFeedback = document.getElementById('feedback-popup');
    const fecharPopupFeedback = document.getElementById('fechar-popup-feedback');
    const feedbackLink = popupFeedback.querySelector('.botao-feedback');

    // Substitua esta URL pela URL do seu servidor Render APÓS a implantação
    const SERVER_URL = 'http://localhost:5000/ask'; // Para testar localmente

    // Lógica do Chatbot
    // Adiciona evento para abrir o chat quando o ícone é clicado
    iconeChat.addEventListener('click', () => {
        popupChat.classList.add('aberto');
        iconeChat.style.display = 'none'; // Esconde o ícone
    });

    // Adiciona evento para minimizar o chat
    botaoMinimizar.addEventListener('click', () => {
        popupChat.classList.remove('aberto');
        iconeChat.style.display = 'flex'; // Exibe o ícone novamente
    });

    // Função para adicionar mensagens ao chat
    function adicionarMensagem(data, remetente) {
        const divMensagem = document.createElement('div');
        divMensagem.classList.add('mensagem');
        if (remetente === 'usuario') {
            divMensagem.classList.add('usuario-mensagem');
        } else {
            divMensagem.classList.add('bot-mensagem');
        }

        divMensagem.textContent = data.text;
        conversaContainer.appendChild(divMensagem);

        if (data.image_url) {
            const imgElement = document.createElement('img');
            imgElement.src = data.image_url;
            imgElement.classList.add('mensagem-imagem');
            conversaContainer.appendChild(imgElement);
        }

        conversaContainer.scrollTop = conversaContainer.scrollHeight;
    }

    // Função para enviar a pergunta para o backend
    async function enviarPergunta() {
        const pergunta = inputUsuario.value.trim();
        if (pergunta === '') {
            return;
        }

        adicionarMensagem({ text: pergunta }, 'usuario');
        inputUsuario.value = '';

        try {
            const response = await fetch(SERVER_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: pergunta }),
            });

            if (!response.ok) {
                throw new Error('Erro na comunicação com o servidor.');
            }

            const data = await response.json();
            
            setTimeout(() => {
                adicionarMensagem(data, 'bot');
            }, 500);

        } catch (error) {
            console.error("Erro ao enviar pergunta:", error);
            adicionarMensagem({ text: "Desculpe, não consigo me conectar. Tente novamente mais tarde." }, 'bot');
        }
    }

    botaoEnviar.addEventListener('click', enviarPergunta);
    inputUsuario.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            enviarPergunta();
        }
    });

    // Lógica do Pop-up de Feedback
    // A cada 30 segundos (30000 ms), o pop-up aparece
    setTimeout(() => {
      popupFeedback.style.display = 'flex';
    }, 30000);

    // Adiciona evento para fechar o pop-up de feedback
    fecharPopupFeedback.addEventListener('click', () => {
      popupFeedback.style.display = 'none';
    });

    // IMPORTANTE:
    // Você precisa criar seu próprio formulário online (Google Forms, Typeform, etc.)
    // e colocar o link dele aqui!
    // Exemplo: https://docs.google.com/forms/d/e/1FAIpQLSc-W...
    feedbackLink.href = "[URL_DO_SEU_FORMULARIO_AQUI]";

});