setwd("C:/Users/erick/Documents/Universidad/Sexto semestre/Proyecto IA")

########################## Importar librerias ##########################
library(moments)
library(ggplot2)
library(reshape2)
library(igraph)

########################## Funciones ##########################
fnModa <- function(x) {
  ux <- unique(x)  # Encuentra los valores únicos en x
  ux[which.max(tabulate(match(x, ux)))]  # Encuentra el valor más frecuente
}

# Función de probabilidad Gaussiana
fnGaussiana <- function(x, mu, s2) {
  A <- 1 / sqrt(2 * pi * s2)
  B <- -0.5 * ((x - mu)^2 / s2)
  fg <- A * exp(B) # vectorial
  return(fg)
}

# Función de probabilidad Gumbel
fnGumbel <- function(x, mu, beta) {
  z <- (x - mu) / beta
  fgum <- (1 / beta) * exp(-(z + exp(-z)))
  return(fgum)
}

# Función de probabilidad Exponencial
fnExponencial <- function(x, lamda) {
  fe <- lamda * exp(-lamda * x)
  return(fe)
}

calcular_priori <- function(data, columna){
  #Obtener posibles valores (en este caso es 1 y 0)
  clases <- unique(data[[columna]])
  clases <- clases[order(clases)]
  
  #Para cada posible valor, se aplica la funcion para obtener la prob.
  priori <- sapply(clases, function(clase){
    sum(data[[columna]] == clase) / nrow(data)
  })
  
  #Se incluyen sus nombres para una mejor visualizacion
  names(priori) <- clases
  return(priori)
}


# Función para estadisticos
calcular_estadisticos <- function(train_data) {
  nc <- length(unique(train_data[[ncol(train_data)]])) # Número de clases
  nvariables <- ncol(train_data) - 1
  estadisticos <- matrix(0, nvariables, nc * 3) # Reservar espacio en memoria
  
  # Obtener los estadísticos
  for (i in 0:(nc - 1)) { # Clases
    for (j in 1:nvariables) { # Variables
      train_variableK <- train_data[train_data[[ncol(train_data)]] == i, j]
      
      # A partir de estos 3 se puede calcular lo necesario para las funciones de probabilidad
      media <- mean(train_variableK, na.rm = TRUE) # Media
      moda <- fnModa(train_variableK) # Moda
      varianza <- var(train_variableK, na.rm = TRUE) # Varianza
      
      estadisticos[j, (1 + 3 * i):(3 + 3 * i)] <- c(media, moda, varianza)
    }
  }
  
  return(estadisticos)
}

# Naive Bayes Gaussiano
fnPredictNBG <- function(test_data, estadisticos, probPriori) {
  nvariables <- ncol(test_data) - 1
  predicciones <- integer(nrow(test_data))
  prob_posteriori <- numeric(length(probPriori))
  
  for (k in 1:nrow(test_data)) {  # Filas
    for (i in 1:length(probPriori)) {  # Clases
      prob_posteriori[i] <- log(probPriori[i])
      for (j in 1:nvariables) {  # Columnas (o variables)
        x <- test_data[k, j]
        
        # Funcion de probabilidad Gaussiana
        media <- estadisticos[j, 3 * (i - 1) + 1]
        var <- estadisticos[j, 3 * (i - 1) + 3]
        
        prob_posteriori[i] <- prob_posteriori[i] + log(fnGaussiana(x, media, var))
        
      }
    }
    predicciones[k] <- which.max(prob_posteriori) - 1 
  }
  
  return(predicciones)
}


# Naive Bayes Mixto
fnPredictNBMix <- function(test_data, estadisticos, probPriori) {
  nvariables <- ncol(test_data) - 1
  predicciones <- integer(nrow(test_data))
  prob_posteriori <- numeric(length(probPriori))
  
  for (k in 1:nrow(test_data)) {  # Filas
    for (i in 1:length(probPriori)) {  # Clases
      prob_posteriori[i] <- log(probPriori[i])
      for (j in 1:nvariables) {  # Columnas (o variables)
        x <- test_data[k, j]
        col_name <- colnames(test_data)[j]
        
        # Funcion de probabildad Gaussiana
        if (col_name %in% c('age', 'Fedu', 'freetime', 'goout')) { 
          media <- estadisticos[j, 3 * (i - 1) + 1]
          var <- estadisticos[j, 3 * (i - 1) + 3]
          
          prob_posteriori[i] <- prob_posteriori[i] + log(fnGaussiana(x, media, var))
          
        } else if (col_name %in% c('studytime', 'famrel', 'health')) {  # Funcion de probabildad Gumble
          moda <- estadisticos[j, 3 * (i - 1) + 2]
          var <- estadisticos[j, 3 * (i - 1) + 3]
          std <- sqrt(var)
          beta <- std * 0.78
          
          prob_posteriori[i] <- prob_posteriori[i] + log(fnGumbel(x, moda, beta))
          
        } else {  # Funcion de probabildad Exponencial
          media <- estadisticos[j, 3 * (i - 1) + 1]
          lambda <- 1 / media
          
          prob_posteriori[i] <- prob_posteriori[i] + log(fnExponencial(x, lambda))
        }
      }
    }
    predicciones[k] <- which.max(prob_posteriori) - 1 
  }
  
  return(predicciones)
}


#Naive Bayes Multinomial
prediccionNBM <- function(datos, priori, probsCondicionales, nombresColumnas){
  logAprobado <- log(priori["1"])
  logReprobado <- log(priori["0"])
  
  for(columna in nombresColumnas){
    valor <- as.character(datos[[columna]])  # Se usara como indice
    logAprobado <- logAprobado + log(probsCondicionales[[columna]][valor,"1"])
    logReprobado <- logReprobado + log(probsCondicionales[[columna]][valor,"0"])
  }
  inferencia <- ifelse(logAprobado>logReprobado,1,0)
  return(inferencia)
}

########################## Lectura de datos ##########################
DF <- read.csv("student-mat.csv", sep = ";")

DF <- subset(DF, select = c(age,Medu,Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3))

########################## ANTES DE LA TAREA DE ML ########################################
############## Discretización de variables (age, absences y G3) ###########################

##### G3

par(mfrow = c(1,2))

barplot(table(DF$G3), main = "Antes", xlab = "Valores", ylab = "Frecuencia")

DF$G3Disc <- ifelse(DF$G3 >= 10, 1, 0)

barplot(table(DF$G3Disc), main = "Después", xlab = "Valores", ylab = "Frecuencia")

mtext("Discretización de G3", outer = TRUE, line = -2, cex = 1.5)

########################## TAREA DE MACHINE LEARNING: NAIVE BAYES ##########################

#Paso 1. Preprocesamiento: Eliminar variables dependientes y no discretizadas.

DF$G3 <- DF$G3Disc

DF <- DF[,!names(DF) %in% c("G1", "G2","ageDisc","G3Disc", "absencesDisc", "Walc", "Medu")]

print(head(DF))

#### Division del dataset
  
set.seed(2820)

train_index <- sample(1:nrow(DF), 0.7*nrow(DF))

train_data <- DF[train_index,]
test_data <- DF[-train_index,]



############## Naive Bayes A (Gaussiano) ##############
### Paso 3 A. Aprendizaje del Naive Bayes 

## Calcular probabilidades a priori (son las mismas para todos los NB)
colEtiqueta <- colnames(train_data)[ncol(train_data)]
priori <- calcular_priori(train_data, colEtiqueta)

## Calcular estadisticos 

estadisticosA <- calcular_estadisticos(train_data)


#Paso 4. Calcular la precision del modelo aprendido
prediccionesA <- fnPredictNBG(test_data, estadisticosA, priori)

mConfA <- table(prediccionesA, test_data$G3)

rendimientoA <- sum(prediccionesA==test_data$G3) / nrow(test_data)
#rendimientoA




######## Naive Bayes B (Mixto)
### Paso 3 A. Aprendizaje del Naive Bayes 

## Calcular estadisticos 
estadisticosB <- calcular_estadisticos(train_data)


#Paso 4. Calcular la precision del modelo aprendido
prediccionesB <- fnPredictNBMix(test_data, estadisticosB, priori)

mConfB <- table(prediccionesB, test_data$G3)

rendimientoB <- sum(prediccionesB==test_data$G3) / nrow(test_data)
#rendimientoB



######## Naive Bayes C (Multinomial)

#### Discretizacion de variables age y absences #####

## Age
cortesAge <- seq(min(DF$age), max(DF$age), length.out = 6)

train_dataM <- train_data
test_dataM <- test_data

train_dataM$ageDisc <- cut(train_dataM$age, breaks = cortesAge, labels = c(1,2,3,4,5), include.lowest = TRUE)
test_dataM$ageDisc <- cut(test_dataM$age, breaks = cortesAge, labels = c(1,2,3,4,5), include.lowest = TRUE)


par(mfrow = c(1,2))
barplot(table(train_dataM$age), main = "Antes", xlab = "Valores", ylab = "Frecuencia")
barplot(table(train_dataM$ageDisc), main = "Después", xlab = "Valores", ylab = "Frecuencia")
mtext("Discretización de age", outer = TRUE, line = -2, cex = 1.5)

## Absences
cortesAsences <- seq(min(DF$absences), max(DF$absences), length.out = 6)

train_dataM$absencesDisc <- cut(train_dataM$absences, breaks = cortesAsences, labels = c(1,2,3,4,5), include.lowest = TRUE)
test_dataM$absencesDisc <- cut(test_dataM$absences, breaks = cortesAsences, labels = c(1,2,3,4,5), include.lowest = TRUE)


par(mfrow = c(1,2))
barplot(table(train_dataM$absences), main = "Antes", xlab = "Valores", ylab = "Frecuencia")
barplot(table(train_dataM$absencesDisc), main = "Después", xlab = "Valores", ylab = "Frecuencia")
mtext("Discretización de absences", outer = TRUE, line = -2, cex = 1.5)



train_dataM$age <- train_dataM$ageDisc
train_dataM$absences <- train_dataM$absencesDisc

test_dataM$age <- test_dataM$ageDisc
test_dataM$absences <- test_dataM$absencesDisc

train_dataM <- train_dataM[,!names(train_dataM) %in% c("ageDisc","absencesDisc")]
test_dataM <- test_dataM[,!names(test_dataM) %in% c("ageDisc","absencesDisc")]


### Paso 3 A. Aprendizaje del Naive Bayes 

## Calcular probs condicionales
colAtributos <- colnames(train_dataM)[-ncol(train_dataM)]

estadisticosC <- sapply(colAtributos, function(atributo){
  #print(atributo)
  frecuenciasAtributo <- table(train_dataM[[atributo]], train_dataM[[colEtiqueta]]) + 1 #Para evitar 0's
  probsAtributo <- prop.table(frecuenciasAtributo, margin = 2) #Margin 2 es para que sea por columnas
  return(probsAtributo)
})


#Paso 4. Calcular la precision del modelo aprendido
prediccionesC <- apply(test_dataM[, -ncol(test_dataM)], 1, prediccionNBM, priori, estadisticosC, colAtributos)


mConfC <- table(prediccionesC, test_dataM$G3)

rendimientoC <- sum(prediccionesC==test_dataM$G3) / nrow(test_dataM)
#rendimientoC



######## Ensamble de Naive Bayes

# Unir los vectores en una matriz
matrizPredicciones <- cbind(prediccionesA, prediccionesB, prediccionesC)


# Calcular la moda de cada fila
prediccionesFinal <- apply(matrizPredicciones, 1, fnModa)

rendimientoFinal <- sum(prediccionesFinal==test_data$G3) / nrow(test_data)
mConfFinal <- table(prediccionesFinal, test_dataM$G3)

mConfA
rendimientoA
mConfB
rendimientoB
mConfC
rendimientoC
mConfFinal
rendimientoFinal
