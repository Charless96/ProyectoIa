setwd("C:/Users/erick/Documents/Universidad/Sexto semestre/6 Ciencia Datos R")

### SEMILLAS ###

## 80 - 20
#68 -> 83%
#515 -> 84%
#5219 -> 86%

## 70 - 30
#68 -> 80%
#2820 -> 84%

set.seed(68)

########################## Importar librerias ##########################
library(moments)
library(ggplot2)
library(reshape2)
library(igraph)

########################## Lectura de datos ##########################
DF <- read.csv("student-mat.csv", sep = ";")

DF <- subset(DF, select = c(age,Medu,Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3))
########################## ANTES DE LA TAREA DE ML ########################################
############## Discretización de variables (G3) ###########################

## G3

DF$G3 <- ifelse(DF$G3 >= 10, 1, 0)

########################## TAREA DE MACHINE LEARNING: NAIVE BAYES ##########################

#Paso 1. Preprocesamiento: Eliminar variables dependientes y no discretizadas.

DF <- DF[,!names(DF) %in% c("G1", "G2","ageDisc","G3Disc", "absencesDisc", "Dalc", "Medu")]

### Paso 2. Separación del dataset (80 - 20)
train_index <- sample(1:nrow(DF), 0.8*nrow(DF))
train_data <- DF[train_index,]
test_data <- DF[-train_index,]

### Paso 3 A. Aprendizaje del Naive Bayes Classifier

## Calcular las probabilidades de cada clase (Aprobado/Reprobado)
calcular_priori <- function(data, columna){
  #Obtener posibles valores (en este caso es 1 y 0)
  clases <- unique(data[[columna]])
  
  #Para cada posible valor, se aplica la funcion para obtener la prob.
  priori <- sapply(clases, function(clase){
    sum(data[[columna]] == clase) / nrow(data)
  })
  
  #Se incluyen sus nombres para una mejor visualizacion
  names(priori) <- clases
  return(priori)
}

colEtiqueta <- colnames(train_data)[ncol(train_data)]
priori <- calcular_priori(train_data, colEtiqueta)

#Calcular la media y la varianza de cada atributo
colAtributos <- colnames(train_data)[-ncol(train_data)]

datosAtributos <- vector("list", length(colAtributos))
names(datosAtributos) <- colAtributos


logicosAprobados <- train_data[[colEtiqueta]] == 1
logicosReprobados <- train_data[[colEtiqueta]] == 0

for(atributo in colAtributos){
  M0 <- mean(train_data[[atributo]][logicosReprobados])
  M1 <- mean(train_data[[atributo]][logicosAprobados])
  
  V0 <- var(train_data[[atributo]][logicosReprobados])
  V1 <- var(train_data[[atributo]][logicosAprobados])
  
  datosAtributos[[atributo]] <- matrix(data = c(M0,V0,M1,V1), nrow = 2, ncol = 2,
                                           dimnames = list(c("Media", "Varianza"), c(0,1)))
}

#En este punto ya tengo la media y varianza por atributo y por clase, ahora queda realizar las predicciones

#Consulta del naive bayes

prediccion <- function(datos, priori, datosAtributos, nombresColumnas){
  posterioriAprobado <- log(priori["1"])
  posterioriReprobado <- log(priori["0"])
  
  for(columna in nombresColumnas){
    #valor <- as.character(datos[[columna]])  # Se usara como indice
    
    mediaApr <- datosAtributos[[columna]]["Media",'1']
    varianzaApr <- datosAtributos[[columna]]["Varianza",'1']

    mediaRep <- datosAtributos[[columna]]["Media",'0']
    varianzaRep <- datosAtributos[[columna]]["Varianza",'0']
    
    muestra <- datos[[columna]]
    
    condicionalAprobado <- log(1 / sqrt(2 * pi * varianzaApr)) - ((muestra - mediaApr)^2 / (2 * varianzaApr))
    condicionalReprobado <- log(1 / sqrt(2 * pi * varianzaRep)) - ((muestra - mediaRep)^2 / (2 * varianzaRep))
    
    posterioriAprobado <- posterioriAprobado + condicionalAprobado
    posterioriReprobado <- posterioriReprobado + condicionalReprobado
  }
  inferencia <- ifelse(posterioriAprobado>posterioriReprobado,1,0)
  return(inferencia)
}

# Obtener las predicciones
predicciones <- apply(test_data[, -ncol(test_data)], 1, prediccion, priori, datosAtributos, colAtributos)

#Evaluar el NBC mediante la matriz de confusion
table(predicciones, test_data$G3)

#Paso 5. Calcular la precision del modelo aprendido

rendimiento <- sum(predicciones==test_data$G3) / nrow(test_data)
rendimiento




