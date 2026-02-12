# Roteiro de Apresentacao em Video

## Classificacao de Doencas Pulmonares - Chatbot Multimodal

**Tech Challenge 6IADT - Fase 4 | FIAP Pos-Tech**

**Duracao estimada: 10-15 minutos**

---

## CENA 1 - Abertura e Introducao (1-2 min)

**[Tela: Slide de titulo ou tela inicial do projeto]**

**Fala:**

> "Ola! Nesta apresentacao vou demonstrar o projeto da Fase 4 do Tech Challenge - um chatbot multimodal para classificacao de doencas pulmonares a partir de imagens de raio-X.
>
> O sistema funciona como um assistente inteligente, inspirado no conceito do Jarvis, que combina Deep Learning com processamento de linguagem natural para oferecer uma experiencia multimodal completa.
>
> O modelo utilizado e uma ResNet50 treinada com Transfer Learning, capaz de classificar raios-X toracicos em 4 categorias:"

**[Tela: Mostrar a tabela de classes no painel lateral da interface]**

> "Covid-19, Healthy (saudavel), Pneumonia Viral e Pneumonia Bacteriana."

---

## CENA 2 - Visao Geral da Arquitetura (1-2 min)

**[Tela: Diagrama de arquitetura do README ou slide proprio]**

**Fala:**

> "A arquitetura do sistema e composta por:
>
> - **Flask** como backend, servindo a interface e processando as requisicoes
> - **ResNet50** para classificacao das imagens de raio-X, com entrada de 256x256 pixels
> - **ChromaDB** como banco vetorial para o sistema RAG, que fornece informacoes complementares de saude
> - **APIs da OpenAI**: GPT-4o-mini para chat e classificacao de mensagens, Whisper para transcricao de audio e TTS-1 para sintese de voz
>
> Toda a interacao acontece em uma interface web single-page que suporta texto, audio, imagem, video e captura de tela."

---

## CENA 3 - Interface do Chatbot (1 min)

**[Tela: Abrir o navegador em http://localhost:5000]**

**Fala:**

> "Esta e a interface do chatbot. Do lado esquerdo temos a area de chat com os controles de interacao. Do lado direito, o painel informativo mostrando as classes de classificacao e as instrucoes de uso.
>
> Na area inferior temos: o campo de upload de imagem de raio-X, o campo de upload de video, a caixa de texto para mensagens, o seletor de dispositivo de audio, e os botoes de enviar, deletar, microfone e a opcao de voz."

---

## CENA 4 - Demonstracao: Chat por Texto (1-2 min)

**[Tela: Interface do chatbot - campo de texto]**

**Fala:**

> "Vamos comecar com a interacao mais basica: o chat por texto."

**[Acao: Digitar uma pergunta geral, ex: "O que sao doencas pulmonares?"]**

> "O sistema classifica automaticamente a mensagem. Se for uma pergunta de saude, ele consulta o banco ChromaDB para recuperar informacoes relevantes e gerar uma resposta contextualizada com o GPT-4o-mini."

**[Acao: Mostrar a resposta aparecendo no chat]**

> "A resposta aparece com o avatar do bot e inclui informacoes baseadas nos documentos de saude armazenados no sistema RAG."

---

## CENA 5 - Demonstracao: Upload de Raio-X (2-3 min)

**[Tela: Interface do chatbot - secao de upload de imagem]**

**Fala:**

> "Agora vou demonstrar a funcionalidade principal: a classificacao de raio-X por imagem."

**[Acao: Clicar em "Escolher imagem" e selecionar uma imagem de raio-X]**

> "Seleciono uma imagem de raio-X toracico..."

**[Acao: Clicar em "Analisar Raio-X"]**

> "Ao clicar em 'Analisar Raio-X', o sistema executa o seguinte fluxo:
>
> 1. Primeiro, o GPT-4o Vision valida se a imagem e realmente um raio-X toracico
> 2. Em seguida, o modelo ResNet50 classifica a imagem
> 3. O sistema consulta o ChromaDB para buscar informacoes de saude relacionadas
> 4. E retorna o resultado completo"

**[Acao: Mostrar o resultado aparecendo - classe, confianca, probabilidades, informacoes de saude]**

> "Aqui vemos a classificacao com a classe predita, o nivel de confianca, as probabilidades de cada uma das 4 classes e as informacoes complementares de saude. Reparem no aviso de que o resultado e apenas informativo."

---

## CENA 6 - Demonstracao: Pergunta de Follow-up (1 min)

**[Tela: Interface do chatbot - campo de texto]**

**Fala:**

> "Apos uma classificacao, o sistema armazena o contexto por 30 minutos, permitindo perguntas de follow-up."

**[Acao: Digitar "explique mais sobre este diagnostico"]**

> "Ao perguntar sobre o diagnostico, o sistema recupera o contexto da ultima classificacao e gera uma resposta mais detalhada, combinando a informacao do raio-X com dados do ChromaDB."

**[Acao: Mostrar a resposta contextualizada]**

---

## CENA 7 - Demonstracao: Analise de Video (2-3 min)

**[Tela: Interface do chatbot - secao de upload de video]**

**Fala:**

> "O sistema tambem suporta analise de video. Vou carregar um video contendo imagens de raio-X."

**[Acao: Selecionar um video e clicar em "Analisar Video"]**

> "Ao iniciar a analise, uma janela do OpenCV abre mostrando o video em tempo real com overlays de classificacao."

**[Tela: Janela do OpenCV com o video sendo processado]**

> "Podemos ver:
> - Em verde, a classe principal com a porcentagem de confianca
> - Em branco, as probabilidades de todas as 4 classes
> - No canto inferior, o contador de frames
>
> O sistema classifica 1 frame a cada 30 - ou seja, aproximadamente uma classificacao por segundo em videos de 30fps. E possivel pressionar 'q' para encerrar antecipadamente."

**[Acao: Aguardar o video terminar ou pressionar 'q']**

> "Apos o processamento, o resultado agregado aparece no chat com a classe dominante, a confianca media, as estatisticas do video e os detalhes frame a frame."

**[Acao: Mostrar o resultado no chat, clicar em "Ver detalhes por frame"]**

---

## CENA 8 - Demonstracao: Deteccao na Tela (1-2 min)

**[Tela: Abrir uma imagem de raio-X no visualizador de imagens do sistema]**

**Fala:**

> "Outra funcionalidade interessante e a deteccao na tela. Vou abrir uma imagem de raio-X no visualizador de imagens do sistema operacional."

**[Acao: Voltar para o chatbot e digitar "o que tem na minha tela?"]**

> "Ao perguntar 'o que tem na minha tela?', o sistema:
> 1. Minimiza o navegador automaticamente
> 2. Captura a tela inteira
> 3. Detecta que ha um raio-X na imagem
> 4. Classifica automaticamente
> 5. Restaura o navegador e mostra o resultado"

**[Acao: Mostrar o resultado da classificacao]**

> "O resultado aparece no mesmo formato da analise de imagem, com todas as probabilidades e informacoes de saude."

---

## CENA 9 - Demonstracao: Interacao por Audio (1-2 min)

**[Tela: Interface do chatbot - area de botoes]**

**Fala:**

> "O chatbot tambem suporta interacao por audio. Primeiro, seleciono o dispositivo de audio - fone de ouvido ou microfone do computador."

**[Acao: Selecionar o dispositivo de audio no dropdown]**

> "Clico no botao do microfone para iniciar a gravacao. O botao fica vermelho e pulsa enquanto grava."

**[Acao: Clicar no microfone, falar uma pergunta, ex: "Quais sao os sintomas da pneumonia?"]**

> "Agora clico novamente para parar a gravacao. O audio e enviado para a API Whisper da OpenAI, que transcreve a fala em texto."

**[Acao: Clicar no microfone novamente para parar]**

> "A transcricao aparece como mensagem do usuario e a resposta do bot e gerada automaticamente."

**[Acao: Mostrar a resposta]**

> "Se a opcao 'Voz' estiver marcada, o chatbot tambem responde em audio usando o TTS-1 da OpenAI."

**[Acao: Marcar checkbox "Voz" e demonstrar uma interacao com resposta em audio]**

---

## CENA 10 - Detalhes Tecnicos (1-2 min)

**[Tela: Slide ou tela com resumo tecnico]**

**Fala:**

> "Resumindo as tecnologias utilizadas:
>
> - **Modelo de classificacao:** ResNet50 com Transfer Learning, treinado no dataset de raios-X toracicos com 4 classes
> - **Preprocessamento:** Imagens redimensionadas para 256x256 pixels e normalizadas
> - **RAG:** Documentos de saude indexados no ChromaDB com embeddings da OpenAI, chunks de 1000 caracteres com overlap de 500
> - **LLM:** GPT-4o-mini para classificacao de intencao, geracao de respostas e analise de imagens via Vision
> - **Audio:** Whisper para speech-to-text e TTS-1 para text-to-speech
> - **Video:** OpenCV para processamento de frames com amostragem e overlay em tempo real
> - **Captura de tela:** Biblioteca mss com suporte a multi-monitor"

---

## CENA 11 - Encerramento (30 seg)

**[Tela: Interface do chatbot ou slide de encerramento]**

**Fala:**

> "Este projeto demonstra como integrar diferentes modalidades de inteligencia artificial em uma unica interface coesa. O chatbot combina visao computacional, processamento de linguagem natural, sintese de voz e retrieval-augmented generation para criar uma experiencia completa de assistencia em saude pulmonar.
>
> E importante reforcar que o sistema e apenas informativo e nao substitui a avaliacao de um profissional de saude.
>
> Obrigado pela atencao!"

---

## Checklist Pre-Gravacao

Antes de gravar, verifique:

- [ ] Servidor Flask rodando (`python chatbot.py`)
- [ ] Navegador aberto em `http://localhost:5000`
- [ ] Imagens de raio-X de teste prontas (pelo menos 1 de cada classe)
- [ ] Video de raio-X de teste pronto
- [ ] Microfone configurado e funcionando
- [ ] Chave da API OpenAI configurada no `.env`
- [ ] ChromaDB populado (`python create_db.py`)
- [ ] Modelo `.keras` presente na pasta `Departamento_Medico/`

## Dicas de Gravacao

- Fale pausadamente e com clareza
- Aguarde o carregamento das respostas antes de prosseguir
- Mostre o cursor na tela para guiar o espectador
- Se um recurso demorar (ex: classificacao de video), explique o que esta acontecendo enquanto processa
- Teste todas as demonstracoes antes de gravar para evitar erros inesperados
