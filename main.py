from flask import Flask, request, render_template, jsonify, redirect, session
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document  # Import necessário para criar os objetos Document
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI



# Configuração do modelo LLM
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")

template = """
Função: O assistente deve atuar como um vendedor virtual, fornecendo respostas precisas e eficazes para as dúvidas dos clientes, utilizando as informações disponíveis. Ele deve verificar na tabela de dúvidas as respostas recomendadas, consultar a tabela de produtos para listar produtos e preços, e aplicar as promoções conforme a tabela de descontos.

Regras:

Estilo da Resposta:

A resposta deve seguir o estilo das melhores respostas anteriores, garantindo coerência em tom de voz, estrutura e argumentos.
Caso não haja uma resposta exata na tabela de dúvidas, a resposta deve ser elaborada com base no estilo das melhores respostas existentes.
Uso da Tabela de Produtos:

O assistente deve verificar se o produto mencionado pelo cliente está na tabela de produtos.
Se o produto estiver disponível, deve informar seu preço e outras informações relevantes.
Se o produto não estiver na lista, responder educadamente informando que ele não está disponível no momento.
Aplicação de Promoções:

O assistente deve conferir se há descontos ou promoções aplicáveis ao produto na tabela de promoções.
Caso haja uma promoção válida, incluir essa informação na resposta e apresentar o preço atualizado com o desconto.
Consulta à Tabela de Dúvidas:

Antes de formular a resposta, o assistente deve verificar se há uma resposta padrão para a dúvida do cliente.
Se houver, a resposta deve ser idêntica ou muito semelhante à melhor resposta registrada.
Se não houver uma resposta exata, o assistente deve construir uma resposta seguindo o tom e a estrutura das melhores respostas.
Respostas Sempre em Português do Brasil.

O texto deve ser natural, claro e persuasivo, garantindo um atendimento profissional e agradável.

Abaixo está a pergunta que recebi do cliente:
{message}

Aqui está uma lista de melhores respostas de como normalmente respondemos a clientes em cenários semelhantes:
{best_practice}

Os produtos vendidos estão nessa lista:
{tabela_produtos}

Os descontos e promoções que podem ser aplicados estão nessa lista:
{tabela_promocoes}

Aqui esta as mensagens anteriores mandadas pelo cliente e as respostas dadas pelo Chatgpt:
{chat_history}
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Carrega as variáveis de ambiente
load_dotenv()

# Caminhos
faiss_index_path = "faiss_index"
csv_path = "Duvidas.csv"

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessário para usar session
app.config["SESSION_PERMANENT"] = False  # Sessão expira quando o navegador fecha

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop("chat_history", None)  # Remove o histórico da sessão
    return jsonify({"message": "Histórico apagado"})

# Inicializa o índice FAISS
if not os.path.exists(faiss_index_path):
    # Carregue o CSV
    df = pd.read_csv(csv_path)

    # Extraia a coluna desejada
    problemas = df["DUVIDA"].tolist()

    # Converta cada string em um objeto Document
    documents = [Document(page_content=issue) for issue in problemas]

    # Crie embeddings e base FAISS
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)

    # Salve o índice FAISS no disco
    db.save_local(faiss_index_path)
else:
    # Carregue o índice FAISS salvo
    db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")

        # Faça a pesquisa de similaridade
        results_with_scores = db.similarity_search_with_score(query, k=3)
        
        # Defina o limite de similaridade
        similarity_threshold = 0.3

        # Filtre os resultados baseados na pontuação
        results = [
            {"content": doc.page_content, "score": score}
            for doc, score in results_with_scores
        ]

    return render_template("index.html", results=results, query=query)

@app.route("/add", methods=["GET", "POST"])
def add_question():
    message = ""

    if request.method == "POST":
        question = request.form.get("question")
        answer = request.form.get("answer")

        if question and answer:
            # Carregue o arquivo CSV
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                # Caso o arquivo não exista, inicialize um DataFrame vazio
                df = pd.DataFrame(columns=["Question", "Answer"])

            # Adicione a nova pergunta e resposta ao DataFrame
            new_row = {"Question": question, "Answer": answer}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Salve de volta no CSV
            df.to_csv(csv_path, index=False)

            # Atualize o índice FAISS com a nova pergunta
            document = Document(page_content=question)
            db.add_documents([document])
            # Salve o índice FAISS no disco
            db.save_local(faiss_index_path)

            message = "Pergunta e resposta adicionadas com sucesso!"
        else:
            message = "Por favor, preencha todos os campos."

    return render_template("add.html", message=message)

@app.route("/ask", methods=["GET", "POST"])
def ask_question():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")

        # Realize a pesquisa de similaridade
        results_with_scores = db.similarity_search_with_score(query, k=3)  # Apenas os 10 primeiros resultados

        # Prepare os resultados para exibição
        for doc, score in results_with_scores:
            # Encontre a resposta correspondente no CSV
            df = pd.read_csv(csv_path)
            answer = df.loc[df["DUVIDA"] == doc.page_content, "RESPOSTAS"].values[0]
            results.append({"question": doc.page_content, "answer": answer, "score": score})

    return render_template("ask.html", results=results, query=query)

@app.route("/chat")
def home():
    return render_template("ask_chatgpt.html")

@app.route("/ask_chatgpt", methods=["POST"])
def ask_chatgpt():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"response": "Por favor, digite uma pergunta."})

    # Inicializa o histórico de mensagens na sessão, se não existir
    if "chat_history" not in session:
        session["chat_history"] = []

    # Adiciona a pergunta do usuário ao histórico
    session["chat_history"].append({"role": "user", "content": query})

    # Realize a pesquisa de similaridade
    results_with_scores = db.similarity_search_with_score(query, k=3)
    best_practices = []
    # Prepare os resultados para exibição
    for doc, score in results_with_scores:
        # Encontre a resposta correspondente no CSV
        df = pd.read_csv(csv_path)
        answer = df.loc[df["DUVIDA"] == doc.page_content, "RESPOSTAS"].values[0]
        best_practices.append({"question": doc.page_content, "answer": answer, "score": score})

    # Carregar os CSVs
    df_produtos = pd.read_csv("Produtos.csv")
    df_promocoes = pd.read_csv("Promocoes.csv")

    # Converter para string formatada (ex.: JSON)
    produtos = df_produtos.to_csv(index=False)
    promocoes = df_promocoes.to_csv(index=False)

    # Criar o contexto com histórico de mensagens
    chat_context = session["chat_history"]
    print(chat_context)
    # Enviar a conversa completa para o modelo GPT
    gpt_response = chain.run({
        "chat_history": chat_context,  # Adicionando o histórico ao prompt
        "message": query,
        "best_practice": best_practices,
        "tabela_produtos": produtos,
        "tabela_promocoes": promocoes
    })

    # Adiciona a resposta do chatbot ao histórico
    session["chat_history"].append({"role": "assistant", "content": gpt_response})

    return jsonify({"response": gpt_response})

@app.route("/table", methods=["GET"])
def show_table():
    # Carregue o CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Question", "Answer"])  # Caso o CSV não exista ainda

    # Converta o DataFrame para uma lista de dicionários para renderizar no HTML
    table_data = df.to_dict(orient="records")

    return render_template("table.html", table_data=table_data)

@app.route("/delete/<int:index>", methods=["POST"])
def delete_entry(index):
    # Verifique se o CSV existe
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Verifique se o índice é válido
        if 0 <= index < len(df):
            # Remova a entrada do DataFrame
            df.drop(index, inplace=True)
            df.reset_index(drop=True, inplace=True)  # Reindexa após a exclusão

            # Salve o CSV atualizado
            df.to_csv(csv_path, index=False)

            # Recrie o índice FAISS com o novo conjunto de perguntas
            questions = df["Question"].tolist()
            documents = [Document(page_content=question) for question in questions]
            global db  # Atualize a variável global db
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(documents, embeddings)
            # Salve o índice FAISS no disco
            db.save_local(faiss_index_path)

    return redirect("/table")




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
