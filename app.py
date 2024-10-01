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
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=5000, value=3500)
    tavily_k = st.slider("Tavily Search Content", min_value=1, max_value=7, value=2)
    
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
    retriever = TavilySearchAPIRetriever(k=tavily_k, include_raw_content=True, api_key=tavily_api_key)
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
Context içerisinde yer alan haber metinlerinden yola çıkarak yeni ve özgün bir haber oluşturman gerekiyor. Bu haber, tamamen orijinal olmalı ve intihal izlenimi vermemeli. Türkçe dil bilgisine uygun, akıcı ve profesyonel bir üslup kullanılmalı.

BAŞLIK
 
* Başlıkta belirsizlik yarat, net olmayan ifadeler kullan. Başlık en fazla 12 kelime olsun, içeriği doğru yansıtsın. Soru cümlesi kullanma. Pasif yapı kullanma. Her bir başlık haberin farklı yönlerini vurgulayan çeşitli açılardan yaklaşmalı. 5 farklı başlık önerisi sun. Nokta ile ayır. 

SPOT

* Spot, başlıkla uyumlu ve haberin ana detaylarını özetler nitelikte olmalı. 1-2 cümle içinde kim, ne, nerede, ne zaman, nasıl, neden gibi soruların cevaplarını ver. Okuyucunun ilgisini çekecek ancak haberin tamamını açık etmeyecek bir dil kullan. Doğal şekilde anahtar kelimeler içermeli ve SEO uyumlu olmalı.

HABER METNİ

PARAGRAFLAR İÇİN TALİMATLAR

* Metni tamamen özgün hale getir ve intihalden arındır. 
* Paragrafları kısa tut, cümleler 12 kelimeyi geçmesin.
* Kritik noktaları ve özel isimleri bold yap.
* "Ebilecek", "abilecek", "ebilir", "abilir", "mektedir", "maktadır" gibi fiillerden kaçın.  
* Aktif cümle yapıları kullan, pasif yapılardan kaçın. Bu sayede daha dinamik ve doğrudan cümleler oluştur. 
* Haberin akışını sağlamak için "ancak", "dolayısıyla", "buna ek olarak" gibi geçiş kelimelerini kullan. 
* Haberi daha etkili sunabilmek için "dedi", "ifadelerini kullandı", "söyledi", "vurguladı", "aktardı", "diye yazdı", "dile getirdi", "açıkladı", "belirtti", "öne çıkardı", "altını çizdi", "şu sözlere yer verdi", "değindi", "işaret etti", "şunu kaydetti", "gündeme taşıdı" gibi ifadeleri konuşma, beyan ve demeç bölümlerinde sıkça kullan.
* Ara başlıkları büyük harflerle yaz. 
* Gereksiz tekrarlar yapma. Aynı ifadeleri tekrarlama.
*  SEO kurallarına uy. Anahtar kelime yoğunluğuna dikkat et. Meta açıklamaları, başlıklar ve alt başlıkları doğru kullan.
* Metne dışarıdan yorum ya da sonuç ekleme.

Giriş (İlk Paragraf)

* Haberin en önemli detaylarını hızlı ve öz bir şekilde özetle. Kim, ne, nerede, ne zaman sorularına net cevap ver. 
* İlk paragraf 30 kelimeyi geçmesin.

Gelişme (Orta Paragraflar) 

* Haberi detaylandırırken her paragraf kısa (3-4 cümle) ve net olmalı.

Context: {context}
"""

birden_fazla_metinden_haber_prompt = """
Context içerisinde yer alan haber metinlerinden yola çıkarak yeni ve özgün bir haber oluşturman gerekiyor. Bu haber, tamamen orijinal olmalı ve intihal izlenimi vermemeli. Türkçe dil bilgisine uygun, akıcı ve profesyonel bir üslup kullanılmalı.

BAŞLIK
 
* Başlıkta belirsizlik yarat, net olmayan ifadeler kullan. Başlık en fazla 12 kelime olsun, içeriği doğru yansıtsın. Soru cümlesi kullanma. Pasif yapı kullanma. Her bir başlık haberin farklı yönlerini vurgulayan çeşitli açılardan yaklaşmalı. 5 farklı başlık önerisi sun. Nokta ile ayır. 

SPOT

* Spot, başlıkla uyumlu ve haberin ana detaylarını özetler nitelikte olmalı. 1-2 cümle içinde kim, ne, nerede, ne zaman, nasıl, neden gibi soruların cevaplarını ver. Okuyucunun ilgisini çekecek ancak haberin tamamını açık etmeyecek bir dil kullan. Doğal şekilde anahtar kelimeler içermeli ve SEO uyumlu olmalı.
HABER METNİ


* Metinleri tamamen özgün hale getir ve intihalden arındır.  

Birbirlerine harmanla ve metinlere şu talimatları uygula:

* Aynı konudaki birden fazla metni kullanarak zenginleştirilmiş, kapsamlı bir haber oluştur.
Metinlerin ortak noktalarını tespit et ve bu unsurları haberin ana eksenine yerleştir. Farklı metinlerde geçen ek bilgileri ve çeşitli bakış açılarını kullanarak haberin detaylarını genişlet.

Tematik Farklılıkları Kullan

* Metinlerde yer alan farklı tema ve bilgileri vurgulayarak haberi daha derinlemesine incele. Bu sayede haber sadece bir kaynağa bağlı kalmadan, daha zengin ve kapsamlı bir içerik sunar. Ancak temalar arasındaki tutarlılığı korumaya dikkat et.

Bilgilerin Uyumluluğuna Dikkat Et

* Farklı kaynaklardaki bilgilerin uyumlu olmasına özen göster. Tutarsızlıkları fark ettiğinde ya doğrulanmış bilgiyi kullan ya da haberde denge kurarak tüm farklı görüşleri doğru şekilde yansıt. Gerektiğinde farklı perspektifleri bağlayıcı geçişlerle sun

PARAGRAFLAR İÇİN TALİMATLAR

* Metni tamamen özgün hale getir ve intihalden arındır. 
* Paragrafları kısa tut, cümleler 12 kelimeyi geçmesin.
* Kritik noktaları ve özel isimleri bold yap.
* "Ebilecek", "abilecek", "ebilir", "abilir", "mektedir", "maktadır" gibi fiillerden kaçın.  
 * Aktif cümle yapıları kullan, pasif yapılardan kaçın. Bu sayede daha dinamik ve doğrudan cümleler oluştur. 
* Haberin akışını sağlamak için "ancak", "dolayısıyla", "buna ek olarak" gibi geçiş kelimelerini kullan. 
* Haberi daha etkili sunabilmek için "dedi", "ifadelerini kullandı", "söyledi", "vurguladı", "aktardı", "diye yazdı", "dile getirdi", "açıkladı", "belirtti", "öne çıkardı", "altını çizdi", "şu sözlere yer verdi", "değindi", "işaret etti", "şunu kaydetti", "gündeme taşıdı" gibi ifadeleri konuşma, beyan ve demeç bölümlerinde sıkça kullan.
* Ara başlıkları büyük harflerle yaz. 
* Gereksiz tekrarlar yapma. Aynı ifadeleri tekrarlama.
*  SEO kurallarına uy. Anahtar kelime yoğunluğuna dikkat et. Meta açıklamaları, başlıklar ve alt başlıkları doğru kullan.
* Metne dışarıdan yorum ya da sonuç ekleme.

Giriş (İlk Paragraf)

* Haberin en önemli detaylarını hızlı ve öz bir şekilde özetle. Kim, ne, nerede, ne zaman sorularına net cevap ver. İlk paragraf 30 kelimeyi geçmesin.

Gelişme (Orta Paragraflar) 

* Haberi detaylandırırken her paragraf kısa (3-4 cümle) ve net olmalı.
Context: {context}
"""

kose_yazisindan_haber_prompt = """
Context içerisinde yer alan haber metinlerinden yola çıkarak yeni ve özgün bir haber oluşturman gerekiyor. Bu haber, tamamen orijinal olmalı ve intihal izlenimi vermemeli. Türkçe dil bilgisine uygun, akıcı ve profesyonel bir üslup kullanılmalı.

BAŞLIK

Haber metninin özünü yansıtacak, kısa ve çarpıcı bir başlık oluştur. Başlık 8-12 kelimeyi geçmesin. Haber hakkında merak uyandıracak ve okuyucuyu içeriğe yönlendirecek anahtar kelimeler içer. Öne çıkan olayları ve konuları basit, net bir dille ifade et.  Cümleler kısa ve öz olsun, karmaşık yapılardan kaçın. Başlıklarda belirsiz özne kullanarak gizem yarat, doğrudan ve çekici bir mesaj ver.

Haber metninden yola çıkarak EN AZ 5 FARKLI başlık önerisi sun. Her bir başlık haberin farklı yönlerini vurgulayan çeşitli açılardan yaklaşmalı.

SPOT

Spot, başlıkla uyumlu ve haberin ana detaylarını özetler nitelikte olmalı. 1-2 cümle içinde kim, ne, nerede, ne zaman, nasıl, neden gibi soruların cevaplarını ver. Okuyucunun ilgisini çekecek ancak haberin tamamını açık etmeyecek bir dil kullan. Doğal şekilde anahtar kelimeler içermeli ve SEO uyumlu olmalı.

HABER METNİ

XXXXXXX yazarı XXXXXXX XXXXXX "YYYYYYY  YYYYYYYY’ başlıklı köşesinde çok önemli noktalara dikkat çekti. 

Bu köşe yazısını daha etkili sunabilmek için "dedi", "ifadelerini kullandı", "söyledi", "vurguladı", "aktardı", "diye yazdı", "dile getirdi", "açıkladı", "belirtti", "öne çıkardı", "altını çizdi", "şu sözlere yer verdi", "değindi", "işaret etti", "şunu kaydetti", "gündeme taşıdı" ve benzeri ifadeleri çeşitli şekillerde kullanarak habere metnine dönüştür. 

Kritik noktaları ve özel isimleri bold yap.

Aktif cümle yapıları kullan. Pasif yapılardan kaçın. Bu sayede daha dinamik ve doğrudan cümleler oluştur. 

SEO kurallarına uy. Anahtar kelime yoğunluğuna dikkat et. Meta açıklamaları, başlıklar ve alt başlıkları doğru kullan.

Metne dışarıdan yorum ya da sonuç ekleme.

Context: {context}
"""




# New prompt for "HABERİ YENİDEN YAZMA"
haberi_yeniden_yazma_prompt = """
Bu metni intihalden tamamen kurtararak habere dönüştür. Başlık, spot ve haber metni şeklinde. 
Ara başlıkları büyük harflerle yaz. Kritik noktaları ve özel isimleri bold yap. Cümleler 12 kelimeyi geçmesin.
Aktif cümle yapıları kullan, pasif yapılardan kaçın.

BAŞLIK YAZMA

1-Volkan Demirel son noktayı koydu!
2-Adana Demirspor yine eli boş döndü!
3-Volkan kritik teklifi kabul etmedi!
4-Aziz Yıldırım Okan Buruk'la ne konuştu. Ortaya çıktı
5-Fenerbahçe'de Mourinho'yu şok eden ayrılık. Bırakıp gitti
6-Mourinho'dan flaş Ali Koç ve derbi açıklaması. Saygısızlık yapıldı
7-Dursun Özbek Mourinho'ya 'Yatarak' karşılık verdi
8-Hacıosmanoğlu Ali Koç'a küçük dilini yutturdu
9-Acun Ilıcalı Bayramiç'te sessiz sedasız çalışmalara başladı
10-Böyle şike görülmedi. 4-1 yenilmesi gerekiyordu 4-1 yenildi
11-Ebrar Karakurt Rusya'yı ayağa kaldırdı. Yok artık
12-Merkezefendi maça çıkarmadan gönderdi
13-Arda Güler 15 dakikada Ancelotti'yi kurtardı
14-Mustafa Denizli imzayı attı
15-Sergen Yalçın yeni adresini açıkladı
16-Mustafa Sarıgül hastaneye koştu. Serhat Akın'ın odasından ilk videoyu paylaştı
17-Van Bronckhorst maçı sattı mı? Sırrı maçtan önce söylediklerinde saklı
18-Temmuz'da kıyamet kopacak!
19-Yüzde 50 zam geldi. Yarından itibaren geçerli!
20-Emekli zil takıp oynayacak! Bayram ikramiyesi netleşti, Nisan’da hesaplara yatacak
21-Kuyumcular Artık Satın Almıyor. 81 İlde Altın İçin Yeni Uygulama Başlıyor
22-Dolarda gece yarısı vurgununu kim yaptı?
23-Parasını Altında Tutanlara Kötü Haber
24-Altın İthalatında Yıllar Sonra Bir İlk
25-Yarın Sabah Başlıyor. Altın Fiyatlarında Haftalar Sonra Bir İlk Yaşanacak
26-İflas eden ünlü iş insanı hurdacılık yaparken görüntülendi
27-Altın Sahiplerine Soğuk Duş. Hazırlığa Başlayın
28-Merkez Bankası'ndan Yeni Döviz Kararı. Dolar Kurunu Altüst Edecek
29-Koç Ailesine Büyük Darbe. Bir Haftada 150 Milyon Dolar Kaybettiler
30-Yarından sonra fiyatlar değişti. Bir zam daha geldi
31-Eşinin aldattığından şüpheleniyordu. Gizli kamera ile eşini kaydeden adam utancından yerin dibine girdi
32-Perdelerini kapatmayı unutan çiftin rezil olduğu anlar kamerada! Sosyal medyada izlenme rekorları kırdı
33-Kocasıyla ilişkiye girdiği sırada yanlışlıkla canlı yayın açtı, babası dahil 45 kişi izledi
34-Meğer onlardan kurtulmak çok kolaymış: Sivrisineklerden jet hızından kurtulmanın tüyosu belli oldu
35-19 ülkeden 34 kişiyle flört eden gezgin kadın Türkiye'de yaşadıklarını anlattı
36-Çocuğunuz Bu Ayda Doğmuşsa Dahi Olabilir!
37-Bu meşrubatı sakın içmeyin, bağışıklık sistemini paramparça yapıyor! Canan Karatay ısrarla uyardı
38-Sakın Çöpe Atmayın! Değerini Öğrenince Çok Şaşıracaksınız
39-Ünlü türkücü, Türkiye'yi terk etti! İsviçre'de köy hayatı yaşıyor
40-Bilenler Nüfus Müdürlüğü'ne Akın Ediyor. Üstelik Bedava Ve Sadece 5 Dakikada İşlem Hallediliyor
41-T.C Kimlik Numarasını Ezbere Bilenlerin Tamamını İlgilendiriyor
42-Bildiğiniz duaları unutun. Teravih namazında dağıtılan suların üstündeki not görenleri hayrete düşürdü
43-Aziz Yıldırım Galatasaray'ı kurtaracak
44-Fenerbahçe 49 topu kaptı
45-Muslera'nın zamanı doluyor
46-Ali Koç ünlü doktoru duyunca ağzı açık kaldı 

Yukarıdaki 46 örnekteki başlıklar, haberlerde birbirinden bağımsız iki konuyu 3-4 kelimeyle, nokta ile ayırıyor. İlk bölümü öznelerin tepkilerine, hareketlerine veya duygularına odaklıyor Bu şekilde okuyucunun dikkatini çekiyor. Yeni yazdığın haberden bu başlık örneklerine benzer 5 adet dikkat çekici ve belirsiz haber başlığı önerisi sun. Nokta ile ayır.

Context: {context}

"""

# Add selection for choosing the prompt, including the new option "HABERİ YENİDEN YAZMA"
prompt_option = st.radio(
    "Select Prompt:",
    ("SEO Content Generator", "BİR METİNDEN HABER YAZMA", "BİRDEN FAZLA METİNDEN HABER YAZMA", "KÖŞE YAZISINDAN HABER YAZMA", "HABERİ YENİDEN YAZMA")
)

# Set default prompt based on the user's selection
if prompt_option == "SEO Content Generator":
    selected_prompt = seo_content_prompt
elif prompt_option == "BİR METİNDEN HABER YAZMA":
    selected_prompt = bir_metinden_haber_prompt
elif prompt_option == "BİRDEN FAZLA METİNDEN HABER YAZMA":
    selected_prompt = birden_fazla_metinden_haber_prompt
elif prompt_option == "KÖŞE YAZISINDAN HABER YAZMA":
    selected_prompt = kose_yazisindan_haber_prompt
else:
    # If "HABERİ YENİDEN YAZMA" is selected
    selected_prompt = haberi_yeniden_yazma_prompt

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
    manual_context = st.text_area("Enter the context information manually:", height=500)

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
        # For non-SEO prompts, including the new "HABERİ YENİDEN YAZMA"
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
