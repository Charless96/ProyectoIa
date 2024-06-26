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

########################## Visualizaciones gráficas ##########################

###### Grafo de dependencias

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



###### Medidas de asociación (matriz de correlación)
#http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization

cormat <- round(cor(DF),2)
melted_cormat <- melt(get_upper_tri(cormat), na.rm = TRUE)
head(melted_cormat)

ggheatmap  <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlación") +
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
