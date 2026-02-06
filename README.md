# Gerekli K??t??phanelerin Y??klenmesi (E??er y??kl?? de??ilse install.packages ile kurunuz)
library(caret)
library(class)
library(e1071)
install.packages(c("caret", "class", "e1071", "RWeka"))

# ---------------------------------------------------------
# ADIM 1: VER?? Y??KLEME
# ---------------------------------------------------------
# Dosya se??me penceresi a????l??r
print("L??tfen fertilizer_recommendation.csv dosyas??n?? se??iniz...")
dataset <- read.csv(file.choose(), stringsAsFactors = TRUE)

# Veriye ilk bak????
str(dataset)
summary(dataset)

# Hedef de??i??kenin da????l??m??n?? g??rme
print("Hedef De??i??ken Da????l??m??:")
table(dataset$Recommended_Fertilizer)

# ---------------------------------------------------------
# ADIM 2: VER?? ??N ????LEME (Preprocessing)
# ---------------------------------------------------------

# A) Fakt??r D??n??????m??
# stringsAsFactors = TRUE ile y??kledi??imiz i??in ??o??u kategorik oldu.
# Ancak hedef de??i??keni a????k??a fakt??r olarak belirtmek iyidir.
dataset$Recommended_Fertilizer <- as.factor(dataset$Recommended_Fertilizer)

# B) Normalizasyon Fonksiyonu (Min-Max Normalizasyonu)
# KNN gibi algoritmalar say??lar aras??ndaki b??y??k farklardan etkilenir.
# Bu y??zden say??lar?? 0 ile 1 aras??na s??k????t??r??yoruz.
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Sadece say??sal s??tunlar?? se??ip normalize edelim
# (Hedef de??i??ken ve di??er kategorik olanlar?? hari?? tutaca????z)

# Say??sal s??tunlar??n indekslerini veya isimlerini belirleyelim
numeric_cols <- sapply(dataset, is.numeric)

# Normalize edilmi?? bir dataframe olu??turuyoruz
dataset_norm <- as.data.frame(lapply(dataset[, numeric_cols], normalize))

# Kategorik s??tunlar?? (Fakt??rleri) normalize edilmi?? veriye geri ekleyelim
# Say??sal olmayan s??tunlar?? al??yoruz
factor_cols <- dataset[, !numeric_cols]

# Hepsini tek bir temiz veri setinde birle??tiriyoruz
final_data <- cbind(factor_cols, dataset_norm)

# Kontrol edelim
str(final_data)
head(final_data)

print("Veri ??n i??leme tamamland??. final_data kullan??ma haz??r.")


#karar a??ac?? 


# ---------------------------------------------------------
# ADIM 3: VER??Y?? E????T??M VE TEST OLARAK B??LME
# ---------------------------------------------------------
# Sonu??lar??n her ??al????t??r????ta ayn?? ????kmas?? i??in seed belirliyoruz
set.seed(123) 

# caret k??t??phanesini kullanarak %70 e??itim, %30 test indeksi olu??turuyoruz
# Hedef de??i??kenin dengeli da????lmas?? i??in createDataPartition kullan??yoruz
trainIndex <- createDataPartition(final_data$Recommended_Fertilizer, p = 0.7, 
                                  list = FALSE, 
                                  times = 1)

train_data <- final_data[ trainIndex,]
test_data  <- final_data[-trainIndex,]

print(paste("E??itim seti boyutu:", nrow(train_data)))
print(paste("Test seti boyutu:", nrow(test_data)))

# ---------------------------------------------------------
# ADIM 4: RWeka ??LE KARAR A??ACI (J48) MODEL??
# ---------------------------------------------------------
library(RWeka)

print("J48 Karar A??ac?? Modeli E??itiliyor...")

# Modeli kuruyoruz
# Form??l: Recommended_Fertilizer ~ . (Hedef de??i??ken ~ Di??er t??m de??i??kenler)
model_j48 <- J48(Recommended_Fertilizer ~ ., data = train_data)

# Modelin ??zetini ve a??a?? kurallar??n?? g??relim
print("Model ??zeti:")
summary(model_j48)

# ---------------------------------------------------------
# ADIM 5: TAHM??N VE DE??ERLEND??RME
# ---------------------------------------------------------
# Test verisi ??zerinde tahmin yapal??m
predictions_j48 <- predict(model_j48, test_data)

# Sonu??lar?? kar????la??t??ral??m (Confusion Matrix)
print("Confusion Matrix (Hata Matrisi) ve ??statistikler:")
cm_j48 <- confusionMatrix(predictions_j48, test_data$Recommended_Fertilizer)
print(cm_j48)

# Sadece Do??ruluk (Accuracy) oran??n?? yazd??ral??m
accuracy_j48 <- cm_j48$overall['Accuracy']
print(paste("J48 Modeli Do??ruluk Oran??:", round(accuracy_j48, 4)))




# ---------------------------------------------------------
# ADIM 6: KARAR A??ACI G??RSELLE??T??RME
# ---------------------------------------------------------

# Daha g??zel grafikler i??in 'partykit' paketini y??kl??yoruz
if(!require(partykit)) install.packages("partykit")
library(partykit)

# J48 modelini grafik nesnesine d??n????t??r??yoruz
model_party <- as.party(model_j48)

print("A??a?? ??izdiriliyor... (B??y??k bir a??a??sa biraz zaman alabilir)")

# ??izim komutu
# gp = gpar(fontsize = 8) komutu yaz??lar?? biraz k??????lt??r ki ekrana s????s??n
plot(model_party, gp = gpar(fontsize = 8))



# ---------------------------------------------------------
# ADIM 6 (REV??ZE): BUDANMI?? (PRUNED) KARAR A??ACI
# ---------------------------------------------------------

print("Budanm???? model e??itiliyor...")

# control parametresi ile ayarlara m??dahale ediyoruz.
# M = 50: Bir dal??n olu??mas?? i??in ucunda en az 50 veri ??rne??i kalmal??.
# Bu say??y?? art??r??rsan (??rn: 100) a??a?? daha da k??????l??r.
model_j48_pruned <- J48(Recommended_Fertilizer ~ ., 
                        data = train_data, 
                        control = Weka_control(M = 50)) 

# Model ??zetine bakal??m (Daha az yaprak say??s?? g??rmelisin)
print("Budanm???? Model ??zeti:")
summary(model_j48_pruned)

# ---------------------------------------------------------
# YEN??DEN G??RSELLE??T??RME
# ---------------------------------------------------------
library(partykit)

# Tekrar grafi??e d??n????t??r
model_party_pruned <- as.party(model_j48_pruned)

print("Budanm???? a??a?? ??izdiriliyor...")

# ??izim (Yaz?? tiplerini biraz k??????lt??p, a??ac?? ekrana s????d??r??yoruz)
plot(model_party_pruned, 
     gp = gpar(fontsize = 9),
     inner_panel = node_inner,
     ip_args = list(abbreviate = TRUE, id = FALSE)) # ??simleri k??saltarak yer a??ar







#KNN

# ---------------------------------------------------------
# ADIM 7: KNN ALGOR??TMASI (class paketi ile)
# ---------------------------------------------------------
library(class)
library(caret) # Confusion Matrix i??in

# KNN sadece say??sal girdilerle ??al??????r.
# Bu y??zden final_data i??indeki SADECE say??sal s??tunlar?? al??yoruz.
# (Not: Kategorik s??tunlar?? modele katmak i??in dummy variable d??n??????m?? gerekir, 
# ancak ??u an temel KNN yap??yoruz)

# Say??sal s??tunlar?? tespit edelim (Hedef de??i??keni hari?? tutmal??y??z)
# final_data'da hangileri numeric?
numeric_cols_indices <- sapply(final_data, is.numeric)

# E??itim ve Test verilerini X (girdiler) ve Y (hedef) olarak ay??ral??m
# train_data ve test_data'y?? J48 ad??m??nda olu??turmu??tuk, onlar?? kullan??yoruz.

# Girdiler (Sadece say??sal olanlar)
train_x <- train_data[, numeric_cols_indices]
test_x  <- test_data[, numeric_cols_indices]

# Hedef De??i??ken (Etiketler)
train_y <- train_data$Recommended_Fertilizer
test_y  <- test_data$Recommended_Fertilizer

print("KNN Modeli ??al????t??r??l??yor...")

# Modeli E??itme ve Tahmin Etme (KNN'de bu ikisi ayn?? anda olur)
# k=21 se??tik. Bu de??eri de??i??tirerek sonu??lar?? g??zlemleyebilirsin.
prediction_knn <- knn(train = train_x, 
                      test = test_x, 
                      cl = train_y, 
                      k = 10)

# ---------------------------------------------------------
# ADIM 8: KNN PERFORMANS DE??ERLEND??RME
# ---------------------------------------------------------

print("KNN Sonu??lar?? (Confusion Matrix):")
cm_knn <- confusionMatrix(prediction_knn, test_y)
print(cm_knn)

# Do??ruluk oran??n?? yazd??ral??m
accuracy_knn <- cm_knn$overall['Accuracy']
print(paste("KNN Modeli Do??ruluk Oran?? (k=21):", round(accuracy_knn, 4)))









#NAV??A BAyes 
# ---------------------------------------------------------
# ADIM 9: NAIVE BAYES ALGOR??TMASI (e1071 paketi ile)
# ---------------------------------------------------------
library(e1071)
library(caret) # Confusion Matrix i??in

print("Naive Bayes Modeli E??itiliyor...")

# Modeli kuruyoruz
# Naive Bayes, 'final_data' i??indeki hem say??sal (normalize edilmi??) 
# hem de kategorik (factor) t??m de??i??kenleri kullanabilir.
model_nb <- naiveBayes(Recommended_Fertilizer ~ ., data = train_data)

# Modelin detaylar??na bakmak istersen (Her s??n??f i??in olas??l??klar?? g??sterir)
# print(model_nb) 

# ---------------------------------------------------------
# ADIM 10: TAHM??N VE DE??ERLEND??RME
# ---------------------------------------------------------

# Test verisi ??zerinden tahmin yapal??m
# type = "class" diyerek do??rudan s??n??f etiketi (Urea, DAP vb.) al??yoruz.
predictions_nb <- predict(model_nb, test_data, type = "class")

# Sonu??lar?? kar????la??t??ral??m (Confusion Matrix)
print("Naive Bayes Sonu??lar?? (Confusion Matrix):")
cm_nb <- confusionMatrix(predictions_nb, test_data$Recommended_Fertilizer)
print(cm_nb)

# Do??ruluk (Accuracy) oran??n?? yazd??ral??m
accuracy_nb <- cm_nb$overall['Accuracy']
print(paste("Naive Bayes Modeli Do??ruluk Oran??:", round(accuracy_nb, 4)))




# T??m modellerin do??ruluk oranlar??n?? kar????la??t??ral??m
basari_tablosu <- data.frame(
  Model = c("J48 (Karar A??ac??)", "KNN (k=10)", "Naive Bayes"),
  Accuracy = c(accuracy_j48, accuracy_knn, accuracy_nb)
)

print("MODEL KAR??ILA??TIRMA SONU??LARI:")
print(basari_tablosu)















# ---------------------------------------------------------
# FARKLI E????T??M ORANLARININ (SPLIT RATIOS) TEST ED??LMES??
# ---------------------------------------------------------

# Denenecek e??itim oranlar?? listesi
oranlar <- c(0.60, 0.70, 0.80, 0.90)

# Sonu??lar?? saklamak i??in bo?? bir tablo olu??turuyoruz
sonuc_tablosu <- data.frame(
  Egitim_Orani = character(),
  Test_Orani = character(),
  KNN_Dogruluk = numeric(),
  NaiveBayes_Dogruluk = numeric(),
  stringsAsFactors = FALSE
)

print("Modeller farkl?? oranlarda test ediliyor, l??tfen bekleyiniz...")

# D??ng?? Ba??l??yor
for (oran in oranlar) {
  
  # 1. VER??Y?? B??LME (Her turda farkl?? oranla)
  set.seed(123) # Her oran i??in ayn?? rastgeleli??i korumak ad??na
  trainIndex <- createDataPartition(final_data$Recommended_Fertilizer, p = oran, 
                                    list = FALSE, 
                                    times = 1)
  
  train_data_loop <- final_data[trainIndex,]
  test_data_loop  <- final_data[-trainIndex,]
  
  # --- KNN MODEL?? ---
  # KNN i??in say??sal s??tunlar?? ay??r??yoruz
  numeric_cols_loop <- sapply(final_data, is.numeric)
  
  train_x_loop <- train_data_loop[, numeric_cols_loop]
  test_x_loop  <- test_data_loop[, numeric_cols_loop]
  
  train_y_loop <- train_data_loop$Recommended_Fertilizer
  test_y_loop  <- test_data_loop$Recommended_Fertilizer
  
  # KNN Tahmini (k=10 sabit tutuyoruz)
  pred_knn_loop <- knn(train = train_x_loop, test = test_x_loop, cl = train_y_loop, k = 10)
  
  # KNN Do??ruluk Hesaplama
  cm_knn_loop <- confusionMatrix(pred_knn_loop, test_y_loop)
  acc_knn <- cm_knn_loop$overall['Accuracy']
  
  # --- NAIVE BAYES MODEL?? ---
  model_nb_loop <- naiveBayes(Recommended_Fertilizer ~ ., data = train_data_loop)
  pred_nb_loop <- predict(model_nb_loop, test_data_loop, type = "class")
  
  # Naive Bayes Do??ruluk Hesaplama
  cm_nb_loop <- confusionMatrix(pred_nb_loop, test_data_loop$Recommended_Fertilizer)
  acc_nb <- cm_nb_loop$overall['Accuracy']
  
  # Sonu??lar?? Tabloya Ekleme
  yeni_satir <- data.frame(
    Egitim_Orani = paste0("%", oran * 100),
    Test_Orani = paste0("%", (1 - oran) * 100),
    KNN_Dogruluk = round(acc_knn, 4),
    NaiveBayes_Dogruluk = round(acc_nb, 4)
  )
  
  sonuc_tablosu <- rbind(sonuc_tablosu, yeni_satir)
}

# ---------------------------------------------------------
# SONU??LARIN G??STER??LMES??
# ---------------------------------------------------------
print("--- FARKLI E????T??M/TEST ORANLARINA G??RE PERFORMANS ---")
print(sonuc_tablosu)



