# ---------------------------------------------------------
# PROJE: Gübre Önerisi Sınıflandırma Modelleri (J48, KNN, Naive Bayes)
# Yazar: Hüsna
# ---------------------------------------------------------

# Gerekli Kütüphanelerin Yüklenmesi
if(!require(caret)) install.packages("caret")
if(!require(class)) install.packages("class")
if(!require(e1071)) install.packages("e1071")
if(!require(RWeka)) install.packages("RWeka")
if(!require(partykit)) install.packages("partykit")

library(caret)
library(class)
library(e1071)
library(RWeka)
library(partykit)

# ---------------------------------------------------------
# ADIM 1: VERİ YÜKLEME
# ---------------------------------------------------------
# Dosya seçme penceresi açılır
print("Lütfen fertilizer_recommendation.csv dosyasını seçiniz...")
dataset <- read.csv(file.choose(), stringsAsFactors = TRUE)

# Veriye ilk bakış
str(dataset)
summary(dataset)

# Hedef değişkenin dağılımını görme
print("Hedef Değişken Dağılımı:")
table(dataset$Recommended_Fertilizer)

# ---------------------------------------------------------
# ADIM 2: VERİ ÖN İŞLEME (Preprocessing)
# ---------------------------------------------------------

# A) Faktör Dönüşümü
dataset$Recommended_Fertilizer <- as.factor(dataset$Recommended_Fertilizer)

# B) Normalizasyon Fonksiyonu (Min-Max Normalizasyonu)
# KNN gibi algoritmalar sayılar arasındaki büyük farklardan etkilenir.
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Sayısal sütunların indekslerini belirleyelim
numeric_cols <- sapply(dataset, is.numeric)

# Normalize edilmiş bir dataframe oluşturuyoruz
dataset_norm <- as.data.frame(lapply(dataset[, numeric_cols], normalize))

# Kategorik sütunları (Faktörleri) normalize edilmiş veriye geri ekleyelim
factor_cols <- dataset[, !numeric_cols]

# Hepsini tek bir temiz veri setinde birleştiriyoruz
final_data <- cbind(factor_cols, dataset_norm)

print("Veri ön işleme tamamlandı. final_data kullanıma hazır.")

# ---------------------------------------------------------
# ADIM 3: VERİYİ EĞİTİM VE TEST OLARAK BÖLME
# ---------------------------------------------------------
set.seed(123) 

# %70 eğitim, %30 test indeksi oluşturuyoruz
trainIndex <- createDataPartition(final_data$Recommended_Fertilizer, p = 0.7, 
                                  list = FALSE, 
                                  times = 1)

train_data <- final_data[ trainIndex,]
test_data  <- final_data[-trainIndex,]

print(paste("Eğitim seti boyutu:", nrow(train_data)))
print(paste("Test seti boyutu:", nrow(test_data)))

# ---------------------------------------------------------
# ADIM 4: RWeka İLE KARAR AĞACI (J48) MODELİ
# ---------------------------------------------------------
print("J48 Karar Ağacı Modeli Eğitiliyor...")

model_j48 <- J48(Recommended_Fertilizer ~ ., data = train_data)

print("Model Özeti:")
summary(model_j48)

# ---------------------------------------------------------
# ADIM 5: TAHMİN VE DEĞERLENDİRME (J48)
# ---------------------------------------------------------
predictions_j48 <- predict(model_j48, test_data)

print("Confusion Matrix (Hata Matrisi) ve İstatistikler:")
cm_j48 <- confusionMatrix(predictions_j48, test_data$Recommended_Fertilizer)
print(cm_j48)

accuracy_j48 <- cm_j48$overall['Accuracy']
print(paste("J48 Modeli Doğruluk Oranı:", round(accuracy_j48, 4)))

# ---------------------------------------------------------
# ADIM 6: KARAR AĞACI GÖRSELLEŞTİRME VE BUDAMA
# ---------------------------------------------------------
model_party <- as.party(model_j48)
print("Ağaç çizdiriliyor...")
plot(model_party, gp = gpar(fontsize = 8))

# --- Budanmış (Pruned) Karar Ağacı ---
print("Budanmış model eğitiliyor...")
model_j48_pruned <- J48(Recommended_Fertilizer ~ ., 
                        data = train_data, 
                        control = Weka_control(M = 50)) 

model_party_pruned <- as.party(model_j48_pruned)
plot(model_party_pruned, 
     gp = gpar(fontsize = 9),
     inner_panel = node_inner,
     ip_args = list(abbreviate = TRUE, id = FALSE))

# ---------------------------------------------------------
# ADIM 7: KNN ALGORİTMASI
# ---------------------------------------------------------
print("KNN Modeli Çalıştırılıyor...")

numeric_cols_indices <- sapply(final_data, is.numeric)
train_x <- train_data[, numeric_cols_indices]
test_x  <- test_data[, numeric_cols_indices]
train_y <- train_data$Recommended_Fertilizer
test_y  <- test_data$Recommended_Fertilizer

prediction_knn <- knn(train = train_x, 
                      test = test_x, 
                      cl = train_y, 
                      k = 10)

print("KNN Sonuçları (Confusion Matrix):")
cm_knn <- confusionMatrix(prediction_knn, test_y)
print(cm_knn)

accuracy_knn <- cm_knn$overall['Accuracy']
print(paste("KNN Modeli Doğruluk Oranı (k=10):", round(accuracy_knn, 4)))

# ---------------------------------------------------------
# ADIM 8: NAIVE BAYES ALGORİTMASI
# ---------------------------------------------------------
print("Naive Bayes Modeli Eğitiliyor...")

model_nb <- naiveBayes(Recommended_Fertilizer ~ ., data = train_data)
predictions_nb <- predict(model_nb, test_data, type = "class")

print("Naive Bayes Sonuçları (Confusion Matrix):")
cm_nb <- confusionMatrix(predictions_nb, test_data$Recommended_Fertilizer)
print(cm_nb)

accuracy_nb <- cm_nb$overall['Accuracy']
print(paste("Naive Bayes Modeli Doğruluk Oranı:", round(accuracy_nb, 4)))

# ---------------------------------------------------------
# ADIM 9: MODEL KARŞILAŞTIRMA VE FARKLI SPLIT RATIOS
# ---------------------------------------------------------
basari_tablosu <- data.frame(
  Model = c("J48 (Karar Ağacı)", "KNN (k=10)", "Naive Bayes"),
  Accuracy = c(accuracy_j48, accuracy_knn
