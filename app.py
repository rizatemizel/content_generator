import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.document_loaders import WebBaseLoader

# Load environment variables (API Keys etc.)
load_dotenv()

# Streamlit App Title
st.title("Content Generator")

# Sidebar section for API Key inputs and model selection
with st.sidebar:
    st.header("API Key Configuration")
    
    st.header("Model Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=6000, value=4500)
    tavily_k = st.slider("Tavily Search Content", min_value=1, max_value=7, value=3)
    
    # Input fields for API keys
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    groq_api_key = st.text_input("Groq API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")

    # Select between Groq and OpenAI models
    model_provider = st.radio(
        "Select Model Provider:",
        ("Groq", "OpenAI")
    )
    
    # Show different model options based on provider
    if model_provider == "OpenAI":
        model_option = st.selectbox("Choose OpenAI Model:", ["gpt-4o", "gpt-4o-mini"])
        if openai_api_key:
            llm = ChatOpenAI(
                model=model_option,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=openai_api_key
            )
        else:
            st.warning("Please provide OpenAI API Key.")
    
    elif model_provider == "Groq":
        model_option = st.selectbox("Choose Groq Model:", ["llama3-70b-8192", "llama-3.1-70b-versatile"])
        if groq_api_key:
            llm = ChatGroq(
                model=model_option,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=groq_api_key
            )
        else:
            st.warning("Please provide Groq API Key.")
    
# Initialize Retriever with the Tavily API Key
if tavily_api_key:
    retriever = TavilySearchAPIRetriever(k=tavily_k, include_raw_content=True, tavily_api_key=tavily_api_key)
else:
    st.warning("Please provide Tavily API Key.")




# Define all prompts
seo_content_prompt = """
Sana verilen contexti kullanarak istenen konu hakkında SEO içerik üretmen bekleniyor.
Türkçe düzgün kullanılmalı, ve profosyonel bir dili olmalı.

BAŞLIK

Kısa, ilgi çekici ve anahtar kelime içeren bir başlık yaz. 8 kelimeyi geçmeyecek şekilde oluşturulmalıdır.

SPOT

"XXXXXX XXXXXX kimdir? XXXXXX XXXXXX kaç yaşında? XXXXXX XXXXXX evli mi?" kelime grubunu içeren, en fazla 12 kelimelik cümleler yaz. Spot 400 harfi geçmemelidir. Haberle ilgili merak uyandırmalı ama içeriği açık etmemelidir.

SEO GİRİŞ

"XXXXXX XXXXXX kimdir? XXXXXX XXXXXX kaç yaşında? XXXXXX XXXXXX evli mi?" kelime grubunu kullanarak, en az 300 harften oluşan bir giriş yaz. Okuyucuyu çeken, dikkat çekici ve anekdot içeren bir metin olsun.

SEO METNİ

"XXXXXX XXXXXX kimdir? XXXXXX XXXXXX kaç yaşında? XXXXXX XXXXXX evli mi?" sorusuna 700 kelimelik bir yanıt ver. Metni özgünleştir, intihalden arındır. Cümleleri 12 kelimeyi geçmemeli. Paragrafları kısa tutmalısın.

SEO

Anahtar kelimeleri doğal bir şekilde metne yerleştir. Anahtar kelimeleri içeren dikkat çekici başlıklar ve alt başlıklar oluştur. SEO dostu, anahtar kelimeler içeren ve kullanıcıyı çeken bir açıklama yaz. İç ve dış bağlantılar ekleyerek SEO performansını artır. Görseller ekle, alt metinlerinde anahtar kelimeleri kullan. İçeriği güncel tut.

GEÇİŞ

"Buna ek olarak," "ancak," "dolayısıyla," "öte yandan" gibi geçiş kelimelerini kullan.

ALINTILAR

Konuşma ve açıklamalarda "dedi," "vurguladı," "belirtti" gibi ifadeleri sıkça kullan.

METİN DÜZENİ

"XXXXXX XXXXXX kimdir? XXXXXX XXXXXX kaç yaşında? XXXXXX XXXXXX evli mi?" ifadesini en az iki kez tekrar et.

Bu adımlara sadık kalarak metni oluştur.

Context: {context}

Konu: {konu}
"""

bir_metinden_haber_prompt = """
Context içerisinde yer alan haber metnini kullanarak yeni bir haber yaratman bekleniyor. Bu haber özgün görünmeli ve intihal hissi yaratmamalı.
Türkçe düzgün kullanılmalı, ve profosyonel bir haber dili olmalı.

BAŞLIK

8 kelimeyi geçmeyecek şekilde dikkat çekici, merak uyandıran başlıklar oluştur. En az 5 tane öneri hazırla.

SPOT

Haberin ana konusunu kısa ve dikkat çekici bir şekilde özetle. Okuyucunun ilgisini çekecek ama tam olarak içeriği açık etmeyecek bir dil kullan. Haber hakkında ipucu ver, ancak ayrıntıya girme. Anahtar kelimeleri ekleyerek metni SEO uyumlu hale getir.

HABER METNİ

Metni özgünleştir. İntihalden tamamen arındır. Haber formatına sadık kal.

Paragrafları kısa tut. Cümleler 12 kelimeyi geçmemeli. "Ebilecek", "abilecek", "ebilir", "abilir", "mektedir", "maktadır" ve benzeri yüklemler kullanma.

Aktif cümle yapıları kullan. Pasif yapılardan kaçın. Bu sayede daha dinamik ve doğrudan cümleler oluştur. Aynı ifadeleri tekrarlama.

Haberi daha etkili sunabilmek için "dedi", "ifadelerini kullandı", "söyledi", "vurguladı", "aktardı", "diye yazdı", "dile getirdi", "açıkladı", "belirtti", "öne çıkardı", "altını çizdi", "şu sözlere yer verdi", "değindi", "işaret etti", "şunu kaydetti", "gündeme taşıdı" ve benzeri ifadeleri konuşma, beyan, demeç, açıklama gibi metinleri bölerken sık sık kullan.

Cümleler arasında "buna ek olarak", "ancak", "dolayısıyla", "öte yandan" gibi geçiş kelimelerine yer ver.

Kritik noktaları ve özel isimleri **bold** yap.

Paragraflar arasında HABER BAŞLIKLARI kullanarak net ve özgün alt başlıklar oluştur.

SEO kurallarına uy. Anahtar kelime yoğunluğuna dikkat et. Meta açıklamaları, başlıklar ve alt başlıkları doğru kullan.

Metne dışarıdan yorum ya da sonuç ekleme.

Context: {context}
"""

birden_fazla_metinden_haber_prompt = """
Context içerisinde yer alan farklı kaynaklardan derlenmiş haber içeriklerini kullanarak yeni ve özgün bir içerik yaratman bekleniyor.

Farklı kaynaklardan derlenmiş içerikler aynı haberi farklı şekilde yazılmış versiyonları olabileceği gibi, farklı şeylerden bahsediyormuş gibi görünen ama belirli bir bağlamda birbiriyle ilişkili içeriklerde olabilir.

Türkçe düzgün kullanılmalı, ve profosyonel bir haber dili olmalı.

BAŞLIK

8 kelimeyi geçmeyecek şekilde dikkat çekici, merak uyandıran başlıklar oluştur. En az 5 tane öneri hazırla.

SPOT

Haberin ana konusunu kısa ve dikkat çekici bir şekilde özetle. Okuyucunun ilgisini çekecek ama tam olarak içeriği açık etmeyecek bir dil kullan. Haber hakkında ipucu ver, ancak ayrıntıya girme. Anahtar kelimeleri ekleyerek metni SEO uyumlu hale getir.

HABER METNİ

Metni ya da metinleri özgünleştir. Metinleri birbirleriyle harmanla. İntihalden tamamen arındır. Haber formatına sadık kal.

Paragrafları kısa tut. Cümleler 12 kelimeyi geçmemeli. "Ebilecek", "abilecek", "ebilir", "abilir", "mektedir", "maktadır" ve benzeri yüklemler kullanma.

Aktif cümle yapıları kullan. Pasif yapılardan kaçın. Bu sayede daha dinamik ve doğrudan cümleler oluştur. Aynı ifadeleri tekrarlama.

Haberi daha etkili sunabilmek için "dedi", "ifadelerini kullandı", "söyledi", "vurguladı", "aktardı", "diye yazdı", "dile getirdi", "açıkladı", "belirtti", "öne çıkardı", "altını çizdi", "şu sözlere yer verdi", "değindi", "işaret etti", "şunu kaydetti", "gündeme taşıdı" ve benzeri ifadeleri konuşma, beyan, demeç, açıklama gibi metinleri bölerken sık sık kullan.

Cümleler arasında "buna ek olarak", "ancak", "dolayısıyla", "öte yandan" gibi geçiş kelimelerine yer ver.

Kritik noktaları ve özel isimleri **bold** yap.

Paragraflar arasında HABER BAŞLIKLARI kullanarak net ve özgün alt başlıklar oluştur.

SEO kurallarına uy. Anahtar kelime yoğunluğuna dikkat et. Meta açıklamaları, başlıklar ve alt başlıkları doğru kullan.

Metne dışarıdan yorum ya da sonuç ekleme.

Context: {context}
"""

kose_yazisindan_haber_prompt = """
Context içerisinde yer alan köşe yazısını özetleyerek haber içeriğine dönüştürmen bekleniyor. Yazarın yazısına saygı duymalı ve bağlamı asla bozmamalısın. 
Türkçe düzgün kullanılmalı, ve profosyonel bir haber dili olmalı.

BAŞLIK

8 kelimeyi geçmeyecek şekilde dikkat çekici, merak uyandıran başlıklar oluştur. En az 5 tane öneri hazırla.

SPOT

Haberin ana konusunu kısa ve dikkat çekici bir şekilde özetle. Okuyucunun ilgisini çekecek ama tam olarak içeriği açık etmeyecek bir dil kullan. Haber hakkında ipucu ver, ancak ayrıntıya girme. Anahtar kelimeleri ekleyerek metni SEO uyumlu hale getir.

HABER METNİ

XXXXXXX yazarı XXXXXXX XXXXXX "YYYYYYY  YYYYYYYY’ başlıklı köşesinde çok önemli noktalara dikkat çekti.

Bu köşe yazısını daha etkili sunabilmek için "dedi", "ifadelerini kullandı", "söyledi", "vurguladı", "aktardı", "diye yazdı", "dile getirdi", "açıkladı", "belirtti", "öne çıkardı", "altını çizdi", "şu sözlere yer verdi", "değindi", "işaret etti", "şunu kaydetti", "gündeme taşıdı" ve benzeri ifadeleri çeşitli şekillerde kullanarak habere metnine dönüştür.

Kritik noktaları ve özel isimleri **bold** yap.

Aktif cümle yapıları kullan. Pasif yapılardan kaçın. Bu sayede daha dinamik ve doğrudan cümleler oluştur.

SEO kurallarına uy. Anahtar kelime yoğunluğuna dikkat et. Meta açıklamaları, başlıklar ve alt başlıkları doğru kullan.

Metne dışarıdan yorum ya da sonuç ekleme.

Context: {context}
"""






# Add selection for choosing the prompt
prompt_option = st.radio(
    "Select Prompt:",
    ("SEO Content Generator", "BİR METİNDEN HABER YAZMA", "BİRDEN FAZLA METİNDEN HABER YAZMA", "KÖŞE YAZISINDAN HABER YAZMA")
)

# Set default prompt based on the user's selection
if prompt_option == "SEO Content Generator":
    selected_prompt = seo_content_prompt
else:
    selected_prompt = bir_metinden_haber_prompt if prompt_option == "BİR METİNDEN HABER YAZMA" else \
                      birden_fazla_metinden_haber_prompt if prompt_option == "BİRDEN FAZLA METİNDEN HABER YAZMA" else \
                      kose_yazisindan_haber_prompt

# Initialize a variable to store the user-updated prompt
updated_prompt = st.text_area("Modify the prompt as needed:", value=selected_prompt, height=500)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(updated_prompt)

# Initialize a variable to store the context (retrieved documents or provided content)
context_content = ""

# Function to format documents into text and store them
def format_docs(docs):
    global context_content
    context_content = "\n\n".join(doc.page_content for doc in docs)
    return context_content

# Function to load and parse content from URLs
def load_url_content(urls):
    if len(urls) == 0 or (len(urls) == 1 and urls[0] == ''):
        return ""
    loaders = [WebBaseLoader(url.strip()) for url in urls if url.strip()]
    documents = []
    for loader in loaders:
        docs = loader.load()
        documents.extend(docs)
    return "\n\n".join(doc.page_content for doc in documents)

# Define the chain for SEO generation
def create_seo_chain():
    return (
        {"context": RunnablePassthrough(), "konu": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Define the chain for non-SEO generation
def create_non_seo_chain():
    return (
        {"context": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Create options for context source
if prompt_option == "SEO Content Generator":
    option = st.radio(
        "Select the source for context information:",
        ("Tavily Search Results", "Manual Context Input", "Paste URLs")
    )
    user_query = st.text_input("Enter the topic for SEO content:")
else:
    option = st.radio(
        "Select the source for context information:",
        ("Manual Context Input", "Paste URLs")
    )
    user_query = None

# Handling the context options
manual_context = ""
urls = []

if option == "Tavily Search Results" and prompt_option == "SEO Content Generator":
    st.write("Search results will be retrieved based on the topic.")
    
elif option == "Manual Context Input":
    manual_context = st.text_area("Enter the context information manually:" , height=500)

elif option == "Paste URLs":
    urls = st.text_area("Paste up to 5 URLs (separated by commas):").split(',')

# Control when the content generation happens with a button
generate_button = st.button("Generate Content")

# Proceed with content generation only when the button is pressed
if generate_button:
    if prompt_option == "SEO Content Generator":
        if option == "Tavily Search Results":
            if user_query:
                with st.spinner("Generating SEO content..."):
                    context_content = format_docs(retriever.get_relevant_documents(user_query))
                    chain = create_seo_chain()
                    result = chain.invoke({"context": context_content, "konu": user_query})
                    st.subheader("Generated SEO Content:")
                    st.write(result)
            else:
                st.warning("Please enter a topic to generate SEO content.")
        elif option == "Manual Context Input" and manual_context:
            if user_query:
                with st.spinner("Generating SEO content..."):
                    chain = create_seo_chain()
                    result = chain.invoke({"context": manual_context, "konu": user_query})
                    st.subheader("Generated SEO Content:")
                    st.write(result)
            else:
                st.warning("Please enter both context and topic.")
        elif option == "Paste URLs" and urls:
            if user_query:
                with st.spinner("Generating SEO content from URLs..."):
                    context_content = load_url_content(urls)
                    if context_content:
                        chain = create_seo_chain()
                        result = chain.invoke({"context": context_content, "konu": user_query})
                        st.subheader("Generated SEO Content:")
                        st.write(result)
                    else:
                        st.warning("Please enter valid URLs.")
            else:
                st.warning("Please enter URLs and a topic to generate SEO content.")
    else:
        # For non-SEO prompts
        if option == "Manual Context Input" and manual_context:
            with st.spinner("Generating content..."):
                chain = create_non_seo_chain()
                result = chain.invoke({"context": manual_context})
                st.subheader("Generated Content:")
                st.write(result)
        elif option == "Paste URLs" and urls:
            with st.spinner("Generating content from URLs..."):
                context_content = load_url_content(urls)
                if context_content:
                    chain = create_non_seo_chain()
                    result = chain.invoke({"context": context_content})
                    st.subheader("Generated Content:")
                    st.write(result)
                else:
                    st.warning("Please enter valid URLs.")

# Expander to show retrieved or input context (documents or provided content)
with st.expander("Context Details"):
    st.write(context_content)
