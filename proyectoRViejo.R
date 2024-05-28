setwd("C:/Users/erick/Documents/Universidad/Sexto semestre/6 Ciencia Datos R")

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

fnCV <- function(x) {
  sd(x) / mean(x)
}

get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA #diag = TRUE ?
  return(cormat)
}

########################## Lectura de datos ##########################
DF <- read.csv("student-mat.csv", sep = ";")

DF <- subset(DF, select = c(age,Medu,Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3))

########################## Visualizaciones gráficas ##########################

##Grafo de dependencias

# Crear un grafo a partir de la matriz de correlación
corM <- cor(DF)
corM <- get_upper_tri(corM)

# Crear un grafo a partir de la matriz de adyacencia
g <- graph_from_adjacency_matrix(corM, mode = "undirected", weighted = TRUE, diag = FALSE)

# Filtrar los bordes por peso (correlación)
g <- delete_edges(g, E(g)[ E(g)$weight < 0.429 ]) #Valor de spearman

E(g)$label <- round(E(g)$weight,2)

g <- delete_vertices(g, degree(g)==0)

# Dibujar el grafo
plot(g, vertex.label=V(g)$name, edge.width=E(g)$weight, vertex.size=30)

########################## Análisis de los datos ##########################

## Resumen estadístico

resumen <- data.frame(
  Media = numeric(),
  Moda = numeric(),
  Mediana = numeric(),
  Varianza = numeric(),
  DesviacionEstandar = numeric(),
  CoeficienteDeVariacion = numeric(),
  Curtosis = numeric(),
  Sesgo = numeric()
)

#Por cada columna (o variable)
for (var in names(DF)) {
  data <- DF[[var]]
  
  resumen[var,] <- c(
    #Media
    mean(data),
    #Moda
    fnModa(data),
    #Mediana
    median(data),
    #Varianza
    var(data),
    #Distribucion estandar
    sd(data),
    #Coeficiente de variacion
    fnCV(data),
    #Curtosis
    kurtosis(data),
    #Sesgo
    skewness(data)
  )
}

resumen <- round(resumen,2)

print(resumen)

## Medidas de asociación (matriz de correlación)
#http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization

cormat <- round(cor(DF),2)
melted_cormat <- melt(get_upper_tri(cormat), na.rm = TRUE)
head(melted_cormat)

ggheatmap  <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

ggheatmap <- ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))


print(ggheatmap)


########################## ANTES DE LA TAREA DE ML ########################################
############## Discretización de variables (age, absences y G3) ###########################

## G3

par(mfrow = c(1,2))

barplot(table(DF$G3), main = "Antes", xlab = "Valores", ylab = "Frecuencia")


DF$G3Disc <- ifelse(DF$G3 >= 10, 1, 0)

barplot(table(DF$G3Disc), main = "Después", xlab = "Valores", ylab = "Frecuencia")

mtext("Discretización de G3", outer = TRUE, line = -2, cex = 1.5)

## Age
cortesAge <- seq(min(DF$age), max(DF$age), length.out = 6)

DF$ageDisc <- cut(DF$age, breaks = cortesAge, labels = c(1,2,3,4,5), include.lowest = TRUE)
DF$ageDisc

par(mfrow = c(1,2))
barplot(table(DF$age), main = "Antes", xlab = "Valores", ylab = "Frecuencia")
barplot(table(DF$ageDisc), main = "Después", xlab = "Valores", ylab = "Frecuencia")
mtext("Discretización de age", outer = TRUE, line = -2, cex = 1.5)

## Absences
cortesAsences <- seq(min(DF$absences), max(DF$absences), length.out = 6)

DF$absencesDisc <- cut(DF$absences, breaks = cortesAsences, labels = c(1,2,3,4,5), include.lowest = TRUE)
DF$absencesDisc

par(mfrow = c(1,2))
barplot(table(DF$absences), main = "Antes", xlab = "Valores", ylab = "Frecuencia")
barplot(table(DF$absencesDisc), main = "Después", xlab = "Valores", ylab = "Frecuencia")
mtext("Discretización de absences", outer = TRUE, line = -2, cex = 1.5)

########################## TAREA DE MACHINE LEARNING: NAIVE BAYES ##########################

#Paso 1. Preprocesamiento: Eliminar variables dependientes y no discretizadas.

DF$age <- DF$ageDisc
DF$absences <- DF$absencesDisc
DF$failures <- DF$failuresDisc
DF$G3 <- DF$G3Disc

DF <- DF[,!names(DF) %in% c("G1", "G2","ageDisc","G3Disc", "absencesDisc", "Dalc", "Medu")]

### Paso 2. Separación del dataset (80 - 20)
train_index <- sample(1:nrow(DF), 0.7*nrow(DF))
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


#Calcular las probs condicionales

colAtributos <- colnames(train_data)[-ncol(train_data)]

probsCodicionales <- sapply(colAtributos, function(atributo){
  #print(atributo)
  frecuenciasAtributo <- table(train_data[[atributo]], train_data[[colEtiqueta]]) + 1 #Para evitar 0's
  probsAtributo <- prop.table(frecuenciasAtributo, margin = 2) #Margin 2 es para que sea por columnas
  return(probsAtributo)
})

#En este punto ya se tienen las probabilidades condicionales para cada atributo.

#Consulta del naive bayes

prediccion <- function(datos, priori, probsCondicionales, nombresColumnas){
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

# Obtener las predicciones
predicciones <- apply(test_data[, -ncol(test_data)], 1, prediccion, priori, probsCodicionales, colAtributos)

#Evaluar el NBC mediante la matriz de confusion
table(predicciones, test_data$G3)

#Paso 5. Calcular la precision del modelo aprendido

rendimiento <- sum(predicciones==test_data$G3) / nrow(test_data)
rendimiento