# Named entity recognition with Bert

BERT Algoritması ile İsim Varlık Tanıma

Emircan Sarıtaş- Feyza Zeynep Salam- Kübra Kurt- Yunus Emre Gündoğmuş- Zehra Karadağ
121517006 - 121517012 - 121517013- 121517004- 121517014

1.Proje amacı: Türk toplumuna yönelik, toplum odaklı bir yol izlenerek Doğal Dil İşleme yöntemlerinin geliştirmesi amaç edinilmiştir.Diğer dillerde NLP’de büyük yol katedilmiş olsa da, şirket CEO’ları bu teknolojinin daha geliştirilecek çok fazla yönünün olduğunu düşünüyor. Türkçe’de şu anda çok fazla veri seti bulunmaması da, Türkçe NLP(Doğal Dil İşleme) henüz taze ve üzerinde çok çalışılma yapılmamış ancak yapılmaya başlanmış olmasından kaynaklanmaktadır. Bizim bu çalışmayı yapmadaki amacımız da NLP dünyasında Türkçe dilinin de gelişimine fayda sağlamak,en yeni teknolojileri deneyimlemektir.
Özetlemek gerekirse; Türkçe’de birçok kullanım alanı olan aynı zamanda çoğu doğal dil işleme modelinin temelinde kullanılan İsim Varlık Tanıma problemini son zamanların popüler  BERT algoritması kullanarak modellemek.


2.Anahtar kelimeler
Transformers: Tensorflow kütüphanesinde BERT i içinde bulunduran paket
BERT: Google’ın diğer birçok algoritma güncellemeleri gibi sorguları daha iyi anlamak ve kullanıcılarına daha doğru sonuçlar sunmak adına geliştirilmiştir. “Bidirectional Encoder Representations from Transformers” ifadelerinin baş harflerinden oluşan BERT algoritması, yapay zeka ve makine öğrenimi teknolojilerini bir arada kullanan bir doğal dil işleme tekniği olarak açıklanabilir.
İsim Varlık Tanıma(NER): Kişi, yer, organizasyon gibi önceden tanımlanmış kategorilerin metin dokümanları üzerinden çıkarılma işlemidir.
Keras: Python’da yazılmış açık kaynaklı bir sinir ağı kütüphanesidir. 
Pytorch: Torch kütüphanesine dayanan açık kaynaklı bir makine öğrenme kütüphanesidir, bilgisayarla görme ve doğal dil işleme gibi uygulamalar için kullanılır.
Pandas: Veri işlemesi ve analizi için Python programlama dilinde yazılmış olan bir yazılım kütüphanesidir.
Numpy: Bilimsel hesaplamaları hızlı bir şekilde yapmamızı sağlayan bir matematik kütüphanesidir.
NLP (Doğal Dil işleme) : Yapay zekâ ve dilbilim alt kategorisidir. Türkçe, İngilizce, Almanca, Fransızca gibi doğal dillerin işlenmesi ve kullanılması amacı ile araştırma yapan bilim dalıdır.
Seqeval: Tahminleri değerlendirmek için kesinlik, geri çağırma ve F1 skor ölçümlerini hesaplamak için python kütüphanesi

3. Veriseti ve Kullanacağımız Model
     3.1 Veriseti
  https://data.mendeley.com/datasets/cdcztymf4k/1   

3.2 Kullanılan Modeller
             3.2.1 NER(İsim Varlık Tanıma)
Bazen varlık toplanması, çıkarılması veya tanımlanması olarak da adlandırılan adlandırılmış   varlık tanıma (NER), metindeki önemli bilgileri (varlıkları) tanımlama ve sınıflandırma görevidir. Bir varlık, sürekli olarak aynı şeyi ifade eden herhangi bir kelime veya kelime dizisi olabilir. Tespit edilen her varlık önceden belirlenmiş bir kategoride sınıflandırılır. Örneğin, bir NER makine öğrenimi (ML) modeli bir metindeki “Amazon” kelimesini algılayabilir ve onu “Şirket” olarak sınıflandırabilir.
Tanımı ilk olarak 1995 yılında MUC-6 (Message Understanding Conference) konferansında yapılmıştır. ENAMEX, TIMEX ve NUMEX olmak üzere 3 temel kategoride tanımlamalar yapılmaktadır.                                                                                                                                   - - - Enamex: Kişi, yer, organizasyon gibi ifaedeleri                                                                                   Numex: Parasal ve yüzdesel ifadeleri                                                                                                Timex: Gün ve tarih gibi zamansal ifadeleri tanımlamak için kullanılmaktadır.                                 İngilizce dilini işlemek için geliştirilmiş olan spacy kütüphanesinde kategorilere daha detaylı olarak yer verilmektedir.

3.2.1.1 Veri Etiketleme Formatı
Raw, IOB, IOB2, BILOU gibi farklı veri etiketleme formatları bulunmaktadır. Aşağıda 
sembollerin açıklaması verilmiştir.
B → Bir varlık isminin başladığını gösterir.
I → Varlık isminin devam ettiğini gösterir.
L → Varlık isminin son kelimesi olduğunu gösterir.
O → Herhangi bir kategoriye ait olmayan varlık isimleri için kullanılır.
U → Tek kelimelik varlık isimlerini tanımlamak için kullanılır.
Aşağıdaki tabloda örnek etiketli veri bulunmaktadır. Veri seti arayanlar CoNLL-2003,     
OntoNote5 verisetlerini inceleyebilirler.

            3.2.1.2 Varlık İsmi Tanıma İçin State of Art Teknikler
Varlık ismi tanıma için en iyi çözümler incelendiğinde skip-gram, glove gibi klasik word embeddinglerin yer almadığını Flair, BERT, ELMO gibi yeni nesil word embedding tekniklerinin yer aldığını görmekteyiz. BLSTM+CRF tabanlı tekniklerin yerini de başka mimariler almaya başlamakta bu çözümler incelendiğinde çok fazla hesaplama gücü gerektirmektedir. BLSTM+CRF tabanlı çözümler computional-cost/accuracy olarak daha uygun gözükmektedir.
https://github.com/sebastianruder/NLP-progress/blob/master/english/named_entity_recognition.md
3.2.1.2.1 BERT
BERT’in açılımı Bidirectional Encoder Representations from Transformers yani Transformatörlerden Çift Yönlü Kodlayıcı Beyanı’dır.  Kısaca anlatmak gerekirse Google aranılan her bir kelimeyi tek bir sorguda işlemek yerine, kullanılan makine öğrenmesi algoritmaları ile daha doğru sonuçlar sağlamak için benzer kelimelere de bakan algoritma güncellemesidir.
BERT, her şeyden önce, Google’ın yeni bilgilerle karşılaştıklarında öğrenen seçkin Makine Öğrenimi Algoritmalarını da kullanıyor.
 BERT’nin bu yeni dil anlama yetenekleri, internet kullanıcılarına aradıklarını net olarak sunacak daha iyi sonuçlar verecektir.
Google’a göre, karmaşık sorgu dizesini kullanan kişilerin nedeni, arama motorlarının başka türlü konuşma sorgularını anlayamama korkusundan kaynaklanıyor. Ancak, BERT Güncellemesi ile Google bu boşluğu kapatmaya çalışacaktır. 
BERT algoritması, geleneksel soldan sağa ve sağdan sola dil işleme modelleri yerine İki Yönlü Dil İşleme özelliğini kullanacaktır. Soldan sağa ve sağdan sola giden yüzeysel çift yönlü dil işlemesinin aksine, BERT her bir kelimenin diğerine olan ilişkisini anlamaya çalışan daha karmaşık bir maskeli dil modeli kullanır.BERT, 102 dilde çok dilli bir model olarak bulunduğundan, çok çeşitli görevler için kullanabilir.
3.2.1.2.1.1 Özet 
BERT hakkında kısa bir özet verelim. BERT'de, bir model önce insan etiketlemesi  gerektirmeyen verilerle önceden eğitilmiştir. Bir kez yapıldığında, önceden eğitilmiş model, girdinin yoğun bir gösterimini çıkarır. KG gibi diğer NLP görevlerini çözmek için, orijinal modelin çıktısına bağlanan sığ bir Deep Learning katmanı ekleyerek modeli değiştiriyoruz. Ardından, modeli göreve özgü veriler ve etiketlerle yeniden eğitiriz.
Kısacası, girdinin yoğun bir temsilini oluşturduğumuz bir eğitim öncesi aşama vardır (aşağıdaki sol diyagram). İkinci aşama, modeli hedef NLP problemini çözmek için SQuAD veya NER gibi göreve özgü verilerle yeniden ayarlar. Biz de NER problemimizi çözmek için elimizdeki veriyle yeniden eğitime başlıyoruz.


3.2.1.2.1.2 Model 
BERT, vektör temsilini oluşturmak için tartıştığımız Transformer kodlayıcısını kullanır. Diğer yaklaşımların aksine, bağlamı yönden ziyade eşzamanlı olarak keşfeder.

3.2.1.2.1.3 Fine-tuning BERT
Model önceden eğitildikten sonra, eğitim öncesi adımda tartıştığımız gibi, herhangi bir NLP görevi veya bir kod çözücü için sığ bir sınıflandırıcı ekleyebiliriz. Biz NER problemiyle ilgilendiğimizden dolayı aşağıdaki şekilde bir gösterim kullanılabilir.


3.2.2.2 NER Nasıl Çalışır?
a.Adlandırılmış bir varlığı algılama
Bir varlığı oluşturan bir kelimenin veya kelime dizesinin algılanmasını içerir. Her kelime bir 
jetonu(token) temsil eder: “Büyük Göller” bir varlığı temsil eden üç jetondan oluşan bir 
dizedir.
b.Varlığı kategorilere ayırma
Varlık kategorilerinin oluşturulmasını gerektirir. Aşağıda bazı yaygın varlık kategorileri verilmiştir:
Kişi:   Örneğin, Mustafa Kemal Atatürk, Audrey Hepburn, David Beckham
Organizasyon: Örneğin, Marmara Üniversitesi, Greenpeace, Oxford Üniversitesi
Zaman: Örneğin, 2006, 16:34, 02:00
Yer: Örneğin, Moda Sahil , Nişantaşı, Maçka Parkı
Sanat eseri: Örneğin, Hamlet, Guernica, Monalisa
 
Görevinize uygun kendi varlık kategorilerinizi oluşturabilir, aynı zamanda belirsizlik veya göreve özgü ontolojilerdeki varlıkların hangi kategorilere ait olduğu hakkında ayrıntılı kurallar sağlayabilirsiniz.

4. Proje adımları

            4.1 Veri Düzenleme
Verinin ilk halinde her bir satırda bir cümle ve etiketleri bulunmaktaydı. Bizim problemimizde her bir kelimenin hangi cümleye ait olduğu ve etiketi bilgisine sahip olunması gerektiğinden, cümleleri kelimelere ayırıp cümle indexlerini de ekleyerek veriyi düzenledik.


Ardından verimizdeki kelime etiketleri BIO Veri Etiketleme Formatına uygun hale getirdik.
            4.2 Model Oluşturma
BERT güçlü bir NLP modelidir, ancak NER veri kümesinde fine-tuning  yapmadan NER için kullanmak iyi sonuçlar vermez. Veri kümesi hazır olduğunda, BERT modelinde fine-tuning yaptık.
Varlığı tespit etmek ve bunları etiketlerde bulunan varlık sınıflarında sınıflandırmak için modelde fine-tuning yapmak üzere bir model oluşturduk.

            4.3 Model Eğitimi
Özet kısmında da bahsedildiği gibi, orijinal modelin çıktısına bağladığımız  sığ bir Deep Learning katmanı ekleyerek modeli değiştiriyoruz. Ardından, modeli göreve özgü veriler ve etiketlerle ikinci kez eğitiyoruz. Buradaki orijinal modelimiz, 35GB ve 44,04,976,662 token büyüklüğünde bir Türk corpus üzerinde önceden eğitilmiş, kasalı bir temel BERT modelidir (BERTürk).BERTürk, Türk toplumuna yönelik, toplum odaklı BERT modelidir.


            4.4 Modelin Test Edilmesi
Son olarak, modelimizin adlandırılmış varlıkları yeni metinde tanımlamasını istiyoruz. Bu cümleyi   verisetimizden rastgele aldık. Önce metni tokenize(parçalarına ayırma) ediyoruz. Sonra cümleyi model üzerinden tahminliyoruz.
Yeni metin:

Tahmin sonucundaki etiketler:

5. Değerlendirme
Bir NER sisteminin çıktı kalitesini değerlendirmek için çeşitli önlemler tanımlanmıştır. Genel ölçümlere Precision, recall ve F1 skoru denir . Bununla birlikte, bu değerlerin nasıl hesaplanacağı konusunda bazı sorunlar devam etmektedir.
Bu istatistiksel önlemler, gerçek bir varlığı tam olarak bulma veya eksiklikle ilgili bariz durumlar için oldukça iyi sonuç verir; ve bir tüzel kişilik bulunmaması için. Bununla birlikte, NER, birçoğu tartışmalı olarak "kısmen doğru" olan ve tam bir başarı veya başarısızlık olarak sayılmaması gereken birçok başka şekilde başarısız olabilir. Örneğin, gerçek bir varlığı tanımlamak, ancak:
istenenden daha az jetonla (örneğin, "John Smith, MD" nin son jetonunu kaçırmak)
istenenden daha fazla jetonla (örneğin, "MD Üniversitesi" nin ilk kelimesi dahil)
bitişik varlıkları farklı bölümlere ayırma (örneğin, "Smith, Jones Robinson" a 2'ye 3 varlık olarak davranma)
tamamen yanlış bir tür atamak (örneğin, kişisel bir adı bir kuruluş olarak adlandırmak)
buna ilgili ancak kesin olmayan bir tür atama (örneğin, "madde" ile "ilaç" veya "okul" ile "kuruluş")
kullanıcının istediği daha küçük veya daha geniş kapsamlı bir varlık olduğunda doğru bir şekilde tanımlamak (örneğin, "James Madison Üniversitesi" nin bir parçası olduğunda "James Madison" ı kişisel bir ad olarak tanımlamak. Bazı NER sistemleri, varlıklar asla üst üste binemez veya yuva yapamaz, bu da bazı durumlarda kişinin keyfi veya göreve özgü seçimler yapması gerektiği anlamına gelir.
Doğruluğu ölçmek için aşırı basit bir yöntem, metindeki tüm belirteçlerin hangi kısmının varlık referanslarının bir parçası olarak (veya doğru türdeki varlıklar olarak) doğru veya yanlış tanımlandığını saymaktır. Bu, en az iki sorundan muzdariptir: Birincisi, gerçek dünyadaki metindeki tokenlerin büyük çoğunluğu varlık adlarının bir parçası değildir, bu nedenle temel doğruluk (her zaman "bir varlık değil" tahmin edin) abartılı bir şekilde yüksektir, tipik olarak>% 90; ve ikincisi, bir varlık adının tam süresini yanlış tahmin etmek doğru şekilde cezalandırılmaz (yalnızca soyadı takip ettiğinde bir kişinin adını bulmak finding doğruluk olarak puanlanabilir).
            6.1 Elde Edilen Skorlar
            
F1-skorumuz %63-67 aralığında olsa bile Modelin Test Edilmesi kısmında da görüldüğü gibi çok da iyi çalışıyormuş gibi görünmüyor.
Biz inanıyoruz ki;
 “There is always a scope of improvement!” .
Bu, BERT kullanarak oluşturduğumuz NER sistemiyle ilgili ilk çalışmamızdı ve bu konuda çalışmaya devam edeceğiz.
Varlıkları daha belirgin bir şekilde kategorilere ayırmak için mümkün olduğunca çok sayıda varlık ekleyerek,
Tıp, politika, eğitim gibi alan adına özgü veri kümeleri için modele fine-tuning yaparak çok daha doğru çalışan bir model ve iyi skorlar elde edebiliriz.
