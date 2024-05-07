<h2>
GI-Assistant (2024/04/22)
</h2>
Bu oshqozon-ichak kasalliklarni ednoskopik tasvirlardan aniqlash dasturi. (Bu to'liq dasturiy taminot emas Faqat kasalliklarni tasvirda belgilab beradi.)


<h3>1. Dataset haqida</h3>
Bu yerda foydalanilgan tasvir maʼlumotlar toʻplami quyidagi veb-saytdan olingan.
</p>
<pre>
Kvasir-SEG Data (Polyp segmentation & detection)
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>

<br>
<h3>2.<b>Kvasir-SEG</b>ma'lumotlar to'plami quyidagi papka tuzilishiga ega.<br></h3>

<pre>
Kvasir-SEG
├─bbox
├─images
└─masks
</pre>

<h3>
3. Tasvirlarni ishlov berishdan oldin tayyorlab olamiz (Preprocessing)
</h3>
Biz tasvirlarni <a href="https://github.com/Sulton21317/GI-assistant/blob/main/resize_images_512x512.py">resize_images_512x512.py</a> dan foydalanib 3 ta <b>train</b>, <b>test</b> va <b>valid</b> ga ajratamiz vv barcha tasvirlarni o'lchamlarini bir xil qilamiz.


Skript quyidagicha tasvirlarni qayta ishlashni amalga oshiradi.<br>
<pre>
1 Tasvirlarni ajratadi va fayllarni maskalaydi (segmentlaydi) <b>train:0.7</b>,<b>valid:0.2 </b> and <b>test:0.1</b>.
2 <b>Kvasir-SEG/images</b> papkasida asl jpg fayllardan 512x512 kvadrat tasvirlar yaratadi.
3 <b>Kvasir-SEG/masks</b> papkasidagi original jpg fayllardan 512x512 kvadrat maska yaratadi (segmentlab oladi) .
4 Oʻlchami oʻzgartirilgan kvadrat tasvirlar va maskalarni kengaytirish uchun 512x512 oʻlchamdagi aylantirilgan, aylantirilgan va aks ettirilgan tasvirlar va maskalar yaratadi.
</pre>
Yaratilgan <by>generated_dataset</b> ma'lumotlar to'plami quyidagi jild tuzilishiga ega.<br>

<pre>
generated_dataset
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<b>O'zgartirilgan tasvir namunalari: generated_dataset/train/images</b><br><br>
<img src="./asset/GastrointestinalPolyp_train_images_sample.png" width="1024" height="auto"><br><br>

<b>Kengaytirilgan niqob namunalari: generated_dataset/train/mask </b><br><br>
<img src="./asset/GastrointestinalPolyp_train_masks_sample.png" width="1024" height="auto"><br>


<h3>
4.Annotatsiya faylini yaratish
</h3>
<h3>
Oshqozon-ichak kasalliklarini aniqlash uchun quyidagilarni amalga oshiramiz.
</h3>

generated_dataset ma'lumotlar to'plamidan oshqozon-ichak yarasi aniqlash va ularni belgilab boshqa tasvirda ko'rsatish
uchun DetectedPolyps papkasini va fayllarini yaratish uchun
<a href="https://github.com/Sulton21317/GI-assistant/blob/main/detect_polyp.py">detect_polyp.py</a> Python skriptini
ishga tushiramiz.
<pre>pyton detect_polyp.py </pre> 
Bu buyruq quyidagi DetectedPolyps papkasini yaratadi, ularda <b>test</b>, <b>train</b> va <b>valid</b> mavjud bo'ladi
<pre>
./DetectedPolyps
├─test
│  └─diseases
├─train
│  └─diseases
└─valid
    └─diseases
</pre>
Masalan, <b>train</b> papkasi juda ko'p jpg tasvir fayllari, qayta ishlangan izohli matn fayllari va izohli papka mavjud
bo'ladi
<br>
<pre>
train
├─diseases
├─flipped_cju0qkwl35piu0993l0dewei2.jpg
├─flipped_cju0qkwl35piu0993l0dewei2.txt
├─flipped_cju0qoxqj9q6s0835b43399p4.jpg
├─flipped_cju0qoxqj9q6s0835b43399p4.txt
...
</pre>
Qayta ishlangan tasvirlar <b>annotated</b> papkasida quyida ko'rsatilgandek chegaralangan to'rtburchak ichida
kasalliklar aniqlangan jpg tasvir fayllari mavjud bo'ladi.
<br>
<br><img src="./asset/train_annotated.png" width="1024" height="auto"><br>

<h3>
Xulosa
</h3>
Bu algoritmlarni video tasvirlarga ishlatish uchun alohida video tasvirlarni frame larga bo'lib olamiz keyin. Ushbu algoritmlar yordamida undan oshqozon-ichak kasalliklarni aniqlaymiz.

