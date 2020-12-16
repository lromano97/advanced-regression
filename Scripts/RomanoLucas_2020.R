### Dataset: https://www.kaggle.com/uciml/electric-power-consumption-data-set


rm(list = ls())
gc()

# install.packages("MASS")
# install.packages("data.table")
# install.packages("corrplot")
# install.packages("psych")
# install.packages("PerformanceAnalytics")
# install.packages("ggplot2")
# install.packages("car")
# install.packages("olsrr")
# install.packages("lmtest")
# install.packages("questionr")
# install.packages("sjPlot")
# install.packages("caret")
# install.packages("vcd")
# install.packages("pROC")

library(MASS)
library(data.table)
library(corrplot)
library(psych)
library(PerformanceAnalytics)
library(ggplot2)
library(car)
library(olsrr)
library(lmtest)
library(questionr)
library(sjPlot)
library(caret)
library(vcd)
library(pROC)

setwd("L:/Data/Projects/regresion-avanzada")

energy <- fread("./Data/household_power_consumption.txt")

summary(energy)

# Date: Fecha en formato dd/mm/yyyy
# Time: Hora en formato hh:mm:ss
# Global_active_power: Consumo energético activo ponderado por minuto de cada casa (en kilowatt)
# Global_reactive_power: Consumo energético reactivo ponderado por minuto de cada casa (en kilowatt)
# Voltage: Voltaje ponderado por minuto (en volt)
# Global_intensity: Intensidad de corriente global ponderada por minuto (en amperes)
# Sub_metering_1: (en watt-hora de energía activa). Correspondiente a una cocina con los siguientes componentes: lavavajillas, horno y microondas.
# Sub_metering_2: (en watt-hora de energía activa). Correspondiente a lavadero con los siguientes componentes: lavarropas, secadora, luz y heladera.
# Sub_metering_3: (en watt-hora de energía activa). Correspondiente a calentador de agua y aire acondicionado

# Transformo las variables numéricas correspondientes.
energy$Date <- as.factor(energy$Date)
energy$Time <- as.factor(energy$Time)
energy$Global_active_power <- as.numeric(energy$Global_active_power)
energy$Global_intensity <- as.numeric(energy$Global_intensity)
energy$Global_reactive_power <- as.numeric(energy$Global_reactive_power)
energy$Voltage <- as.numeric(energy$Voltage)
energy$Sub_metering_1 <- as.numeric(energy$Sub_metering_1)
energy$Sub_metering_2 <- as.numeric(energy$Sub_metering_2)
energy$Sub_metering_3 <- as.numeric(energy$Sub_metering_3)

# Tras hacer la transformación aparecen varias filas que se encuentran en NA y parecen ser las mismas para todos los casos. Es por esto por lo que se proseguirán a remover las mismas del dataset.
summary(energy)

# Para validar que todos los NA corresponden a los mismos registros se prosigue a eliminar los NA basándose en una única columna.
energy <- na.omit(energy, cols="Sub_metering_3")

# Creo una variable categórica para incluir en el análisis. Esta va a estar relacionada con la mediana de Global_active_power y evaluara en alto = 1 o bajo = 0 dependiendo el caso.
avg.consupmtion.mean <- median(energy$Global_active_power)
energy[, avg.consumption := as.factor(ifelse(Global_active_power > avg.consupmtion.mean, 1, 0))]

# Además, se agrega una variable que indica sumariza el consumo de los distintos dispositivos y los clasifica en alto o bajo con el fin de poder hacer el análisis de la regresión logística más fructífero.
energy[, device.consumption := energy$Sub_metering_1 + energy$Sub_metering_2 + energy$Sub_metering_3]
avg.consumption.device <- median(energy$device.consumption)
energy[, avg.consumption.device := as.factor(ifelse(device.consumption > avg.consumption.device, "High", "Low"))]
energy[, device.consumption := NULL]

# Tal como se supuso, todos los NAs observados correspondían a los mismos registros.
summary(energy)

# Creo una partición aleatoria para poder realizar el análisis de los datos con los recursos computacionales disponibles.
set.seed(333)
sample.size <- floor(0.1 * nrow(energy))
sample.energy.idx <- sample(seq_len(nrow(energy)), size = sample.size)
sample.energy <- energy[sample.energy.idx]

##### ANÁLISIS EXPLORATORIO #####

# Obtengo las variables numéricas para poder evaluar la correlación.
numeric.idx <- which(sapply(sample.energy, class) %in% c("numeric", "integer"))
numeric.sample.energy <- subset(sample.energy, select = numeric.idx)

# Obtengo la correlación a través de Spearman y Pearson

lineal.energy.correlation <- cor(numeric.sample.energy)
spearman.energy.correlation <- cor(numeric.sample.energy, y = NULL, use = "everything", method = c("spearman"))

# Dado que las variables no se comportan de manera normal, es necesario aplicar el test de correlación no paramétrico de Spearman y no el de Pearson ya que no cumple el supuesto de normalidad.

# A fines prácticos, dado el alto costo en tiempo de Kendall y que el mismo está diseñado para evaluar variables ordinales se tomara muestra aleatoria mucho más pequeña para poder obtener la correlación.
small.sample.energy.idx <- sample(seq_len(nrow(energy)), size = 50)
small.sample.energy <- energy[small.sample.energy.idx]
small.numeric.idx <- which(sapply(sample.energy, class) %in% c("numeric", "integer"))
numeric.small.sample.energy <- subset(sample.energy, select = small.numeric.idx)
kendall.energy.correlation <- cor(numeric.small.sample.energy, y = NULL, use = "everything", method = c("kendall"))
# TODO: Dejarlo ejecutando el tiempo que sea necesario para análisis


corPlot(numeric.sample.energy, cex = 1.1, main = "Matriz de correlacion")
chart.Correlation(numeric.sample.energy, histogram = TRUE, lm = TRUE, method = "pearson")
# Nuevamente realizamos el grafico de correlación de las variables, pero en este caso utilizando el método Spearman que no considera que las variables se distribuyan de manera normal.
chart.Correlation(numeric.sample.energy, histogram = TRUE, lm = TRUE, method = "spearman")


# Si me basara en la correlación lineal se podría observar una correlación muy fuerte entre Global_active_power (variable target) y Global_intensity por lo cual puede llegar a ser útil realizar una regresión lineal (en un principio) para poder predecir la variable target. 
# Por otro lado se observan otras correlaciones lineales no tan fuertes, pero que valen la pena analizar, que son la que se da por Sub_metering_3 con Global_active_power o de Sub_metering_1 con Global_active_power, aunque esta última es casi despreciable en comparación a las anteriores.
# En este caso, al aplicar Spearman voy a ver si existe correlación entre las variables. Para este dataset, se puede notar una notoria diferencia entre algunas de las correlaciones obtenidas a través de Pearson y las obtenidas a través de Spearman, particularmente para el caso de Sub_metering_1, que en la correlación de Pearson daba un valor cercano a 0.5, pero para Spearman ese valor es mucho más bajo. esto se observa en el grafico obtenido en la sentencia de chart.Correlation(numeric.sample.energy, histogram = TRUE, lm = TRUE, method = "spearman").


# Gráfico de dispersión entre Global_intensity y Global_active_power
ggplot(sample.energy, aes(x=Global_intensity, y=Global_active_power)) +
    geom_point() +
    geom_smooth(method=lm, color="red", fill="#69b3a2", se=T)
# En este grafico se puede ver como claramente las variables Global_intensity y Global_active_power siguen una relación lineal casi perfecta, donde todos los puntos se ubican cercanos a una función lineal que surge como resultado de entrenar un modelo de regresión lineal sobre los datos.

# Gráfico de dispersión entre Sub_metering_3 y Global_active_power
ggplot(sample.energy, aes(x=Sub_metering_3, y=Global_active_power)) +
    geom_point() +
    geom_smooth(method=lm, color="red", fill="#69b3a2", se=T)
# En este caso se puede ver como los valores mínimos de Global_active_power se encuentran sobre una recta, pero en general no parece seguir una correlación lineal. Es importante destacar que, en el análisis previo de correlaciones, esta dio un valor muy bajo en comparación a la anterior comparación.

# Gráfico de dispersión entre Sub_metering_1 y Global_active_power
ggplot(sample.energy, aes(x=Sub_metering_1, y=Global_active_power)) +
    geom_point() +
    geom_smooth(method=lm, color="red", fill="#69b3a2", se=T)
# En este caso, al igual que en el anterior se observa el mismo comportamiento en cuanto a los valores mínimos, pero a medida que los valores de Global_active_power van aumentando la forma de la relación escapa la forma lineal deseada para poder aplicar una regresión lineal simple.

# Combinamos todos los gráficos realizados previamente en uno solo, además incluyendo la combinación entre las variables independientes.
pairs(~Global_active_power + Global_intensity + Sub_metering_3 + Sub_metering_1, data=sample.energy, main="Energy consumption scatterplot")

##### REGRESION LINEAL MULTIPLE #####

# Nota: Como este es un ejemplo didáctico se generará el modelo utilizando todas las variables que dieron dentro de todo una correlación significativa. En el caso de que sea un caso de la vida real, posiblemente se generaría el modelo utilizando únicamente la variable Global_intensity como predictora.

# Genero el modelo de regresión lineal múltiple utilizando todas las variables numéricas disponibles
model.1 <- lm(Global_active_power ~ Global_intensity + Sub_metering_3 + Sub_metering_1 + Voltage + Sub_metering_2 + Global_reactive_power, data = sample.energy)
summary(model.1)
# Al obtener el summary del modelo obtenemos que todas las variables rechazan la hipótesis nula de que el coeficiente beta n es cero para cada una de ellas, por lo cual son significativas para el modelo. 
# La ecuación para este modelo quedaría: -1.075e+00 + 2.382e-01 * Global_intensity + 2.132e-03 * Sub_metering_3 - 3.327e-04 * Sub_metering_1 + 4.456e-03 * Voltage - 5.630e-04 * Sub_metering_2 - 1.760e-01 * Global_reactive_power
# El valor del R cuadrado ajustado es alto, por lo cual el modelo es un buen predictor de la variable Global_active_power y el valor de p-value que evalúa si todos los coeficientes son cero es muy cercano a cero, lo cual es esperable dado que todos los coeficientes también tienen valor de p-value cercano a cero.
# Cada uno de los coeficientes que acompaña a las variables representa cómo se comporta Global_active_power al aumentar o disminuir en una unidad cada una de las demás variables predictoras.

# Genero otro modelo, pero ahora únicamente tomo aquellas variables que tienen una alta correlación lineal con la variable a predecir con el fin de obtener un modelo más parsimonioso.
model.2 <- lm(Global_active_power ~ Global_intensity + Sub_metering_3, data = sample.energy)
summary(model.2)
# En este caso, al obtener el summary, todas las variables predictoras son útiles para el problema y los coeficientes son ligeramente distintos a los vistos en el primer modelo.
# La ecuación para este modelo quedaría: -1.070e-02 + 2.346e-01 * Global_intensity + 2.609e-03 * Sub_metering_3
# Nuevamente el valor de R cuadrado ajustado da alto, pero en este caso hay un ligero decremento en comparación a lo visto en el modelo anterior. El p-value que evalúa si la hipótesis de si los coeficientes son iguales a cero da lo mismo que en el modelo anterior, por lo cual hay un coeficiente que es distinto de cero (en este caso son los tres).

# Si me basará en el valor de R cuadrado ajustado en conjunto con el concepto de parsimonia, terminaría eligiendo el modelo dos por sobre el uno dado que la cantidad de variables utilizadas es mucho menor en este último a pesar de presentar un minúsculo decremento en el valor de R cuadrado.

par(mfrow = c(2,2))
plot(model.1)
# Al observar los gráficos obtenidos se puede notar como:
# 1. El grafico muestra la relación entre los residuos y los valores predichos. A partir del mismo se va a observar si los residuos presentan homocedasticidad. En este caso parecen no presentarla dado que no muestran una estructura aleatoria los distintos puntos observados ya que a medida que se avanza sobre el eje x la varianza de las observaciones va disminuyendo.
# 2. Los residuos no se distribuyen de manera normal. Se puede ver como al principio, los valores que se encuentran desde -4 a -2 rompen con la normalidad de los errores.
# 3. En el tercer grafico (Scale-Location) se puede observar cómo los errores siguen una forma aleatoria.
# 4. En el cuarto grafico se observa la influencia de los residuos en la regresión. En este caso se puede observar cómo los valores que se encuentran entre x=4e-04 y x=6e-04 cercanos a y=-10 tienen mayor influencia en el modelo.


plot(model.2)
# Al observar los gráficos obtenidos se puede cosas muy similares al modelo anterior, como:
# 1. Al igual que el modelo 1 se puede observar cómo se comportan los residuos y los valores predichos. Los residuos no parecen ser homocedasticos dado que la varianza parece no ser constante a lo largo de todas las observaciones, como en el modelo 1, sino que a medida que se avanza sobre el eje x la varianza va disminuyendo.
# 2. Los residuos se distribuyen de la misma forma que los residuos del modelo anterior
# 3. En el tercer grafico (Scale-Location) se puede observar cómo los errores siguen una forma aleatoria, al igual que en el modelo 1.
# 4. En el cuarto grafico se observa la influencia de los residuos en la regresión. En este caso los residuos más influyentes son los que se encuentran más alejados de x=0.


# Obtengo los valores predichos por cada uno de los modelos
predicted.model.1 <- fitted(model.1)
predicted.model.2 <- fitted(model.2)

# Obtengo los residuos de las predicciones realizadas
residuals.model.1 <- residuals(model.1, type="response")
residuals.model.2 <- residuals(model.2, type="response")

# Unifico todos los valores predichos y los residuos en la misma tabla.
energy.final <- cbind(sample.energy, predicted.model.1, residuals.model.1, predicted.model.2, residuals.model.2)

coefficients(model.1)
coefficients(model.2)
# Al ejecutar las sentencias de coefficients se obtienen nuevamente los valores de beta cero y beta uno que previamente fueron mencionados al realizar el summary de cada uno de los modelos.

confint(model.1, level=0.95)
confint(model.2, level=0.95)
# Al obtener el intervalo de confianza de 0.95 para cada uno de los beta de los distintos modelos. Ninguno de los coeficientes tiene en su intervalo el valor cero. Es por esta razón que al generar los modelos se rechaza la hipótesis nula de que el beta n analizado es igual a cero para cada uno de los coeficientes obtenidos (p-value muy bajo visto en el summary del modelo), por lo cual todas las variables predictoras utilizadas son significativas para el modelo.

# A partir de las siguientes líneas de código se podrá observar cómo se comportan las distintas observaciones y cuanto influyen sobre el modelo final mediante la distancia de Cook. Esto va a ser principalmente útil para poder detectar posibles puntos influyentes en el dataset. Se va a considerar como puntos influyentes aquellos valores de distancia de Cook mayor o igual 4/n donde n es la cantidad de observaciones (1.951905e-05).
cook.distance.model.1=cooks.distance(model.1)
plot(cook.distance.model.1, ylab="Distancia de Cook")
abline(h=1.951905e-05)

length(cook.distance.model.1 > 1.951905e-05)

cook.distance.model.2=cooks.distance(model.2)
plot(cook.distance.model.2, ylab="Distancia de Cook")
abline(h=1.951905e-05)

length(cook.distance.model.2 > 1.951905e-05)

# En ambos, si se tomara en cuenta el límite previamente mencionado, la cantidad de valores influyentes es muy grande (casi más de la mitad de los datos). Ahora bien, al ser tan grande la cantidad de observaciones que se están manejando el valor limite pasa a ser extremadamente chico. Visualmente se puede observar que el ese valor está por debajo de una gran masa de puntos en el gráfico. Por esto, se suele tomar como limite el valor 1 en lugar de 4/n, en cuyo caso, no quedaría ninguna observación como influyente para ninguno de los dos modelos.

# https://en.wikipedia.org/wiki/Cook%27s_distance#:~:text=8%20Further%20reading-,Definition,closer%20examination%20in%20the%20analysis.

# Otros gráficos en donde se aplica la distancia de Cook

ols_plot_cooksd_bar(model.1)
ols_plot_cooksd_chart(model.1)

ols_plot_cooksd_bar(model.2)
ols_plot_cooksd_chart(model.2)

# En ambos casos los valores que se obtienen para las distancias de Cook no son mayores a uno, por lo cual no son considerados como observaciones que influencien en el modelo.

# A continuación se prosigue a calcular los residuos studentizados, en ellos se observara aquellos que contengan un valor mayor a 3, los cuales serán considerados como outliers.
student.residual = rstudent(model.2)
length(which(abs(student.residual) > 3))
plot(student.residual, ylab="R studentizado")
# Si me basará en los residuos studentizados hay 2193 observaciones que son consideradas como outlier en el dataset para el modelo. 

# Leverage
# El leverage me permite observar si existe algún punto que genere que mi recta de regresión se desplace con el fin de poder tener en cuenta ese valor.
ols_plot_resid_lev(model.2)
# En este caso se pueden observar varios puntos que tienen leverage en el modelo, no solamente son outliers, sino que también se pueden observar valores que no son outliers, pero poseen un leverage considerable. Es importante tener en cuenta que dada la gran cantidad de observaciones que tiene el modelo el límite o threshold que existe para considerar que un valor tiene leverage es muy bajo (1.951905e-05).

summary(influence.measures(model.2))
# Si obtengo el cálculo de las distintas observaciones influyentes calculadas por distintos métodos, aproximadamente 11.000 observaciones son influyentes para el modelo dos. Es decir, la ecuación de regresión se desplazará con el fin de poder predecir de manera correcta los distintos puntos influyentes.

##### SUPUESTOS ##### 

# Shapiro wilks (Normalidad de los errores)
# Genero una muestra aleatoria de los residuos para poder ejecutar shapiro.test ya que el límite en cantidad de variables es de 5000.
set.seed(333)
residuals.model.1.sample <- sample(seq_len(length(residuals.model.1)), size = 5000)
residuals.model.2.sample <- sample(seq_len(length(residuals.model.2)), size = 5000)

shapiro.test(residuals.model.1.sample) 
# Al aplicar el test de Shapiro para evaluar la normalidad de los errores en el modelo 1 se puede observar cómo se rechaza la hipótesis de que la muestra aleatoria de los errores sigue una distribución normal. 
# Por este motivo, el modelo no podrá utilizarse para poder realizar estimaciones. Una alternativa para poder solucionar el problema de la normalidad de los residuos sería aplicar una transformación de Box y Cox, escalar los valores o algún otro tipo de transformación de las variables.

shapiro.test(residuals.model.2.sample)
# Para el modelo 2 ocurre la misma situación que para el modelo uno (se rechaza la normalidad de los residuos).

# Estadísticas de los residuos (compruebo media = 0)
summary(residuals.model.1)
summary(residuals.model.2)
# En este caso, al realizar el summary de los distintos residuos se puede observar como la media es igual a cero por lo cual los residuos son independientes.

# Test de autocorrelación de los errores (Durbin-Watson)
# Mediante el test de Durbin Watson voy a validar la correlación entre los residuos del modelo para poder ver si hay algún tipo de relación entre los mismos.
set.seed(333)
durbinWatsonTest(model.1)
# Al aplicar Durbin Watson para los residuos del modelo uno da como resultado que los mismos no se encuentran correlacionados (p-value es muy superior a 0.05).

durbinWatsonTest(model.2)
# Para el caso del modelo dos, si bien da que no se rechaza la hipótesis nula (no hay correlación entre los residuos), el p-value es superior al p-value del modelo 1

# gráficos de dispersión Residual vs Predichos
# En los siguientes gráficos se va a poder observar cómo se relacionan los distintos valores predichos con los residuos. De esta forma se podrá observar si existe homocedasticidad de los residuos. La interpretación de los gráficos es similar a la que se realizó en la parte de plot() de los distintos modelos, donde además de este grafico de dispersión se encontraba el grafico QQ-plot de los residuos, como así también los gráficos de la distancia de Cook.
par(mfrow=c(1,1))
attach(sample.energy)
plot(predicted.model.1, residuals.model.1,
     las=2,
     col="blue",
     pch=16,
     xlim=c(-1,10),
     ylim=c(-1,1),
     main="Relacion entre residuales y valores predichos - Modelo 1")
title(ylab="", line=3.3)


par(mfrow=c(1,1))
attach(sample.energy)
plot(predicted.model.2, residuals.model.2,
     las=2,
     col="blue",
     pch=16,
     xlim=c(-1,10),
     ylim=c(-1,3),
     main="Relacion entre residuales y valores predichos - Modelo 2")
title(ylab="", line=3.3)

# En este caso se aplicará además el test para evaluar la homocedasticidad (Breusch Pagan) para verificar lo planteado en el análisis de los gráficos.
bptest(model.1)
bptest(model.2)
# En ambos casos se puede observar cómo se rechaza la hipótesis nula de que el modelo presenta homocedasticidad, para poder solucionar este problema, al igual que el problema de normalidad de los residuos, se puede aplicar una transformación de Box & Cox.

##### EVALUACIÓN DE MODELOS ##### 

# Criterio de información de Akaike
AIC(model.1, model.2)
# Al evaluar el AIC de ambos modelos, el valor obtenido es muy similar, lo cual llevaría a pensar que ambos modelos son igual de buenos. Al momento de elegir un modelo, claramente sería aquel más parsimonioso, que, en este caso, sería el modelo 2 que tiene una variable menos que el modelo 1.

# Realizo una predicción de prueba con ambos modelos
predict(model.1, data.frame(Global_intensity = 0.2, Sub_metering_1 = 0, Sub_metering_3 = 1, Global_active_power = 0.08, Voltage = 239.83, Sub_metering_2 = 0, Global_reactive_power = 0), interval = "prediction")
predict(model.2, data.frame(Global_intensity = 0.2, Sub_metering_1 = 0, Sub_metering_3 = 1, Global_active_power = 0.08), interval = "prediction")
# En ambos casos los valores predichos son relativamente similares, pero en el caso del modelo uno, el valor predicho es más cercano al valor real de Global_active_power.

# A continuación se pasará el modelo uno (el que contiene todas las variables numéricas) a la función stepAIC con el fin de que busque seleccionar de manera automática las variables predictoras que serán necesarias para poder generar un modelo que logre predecir la variable target. Se llevarán a cabo distintos enfoques vistos en clase, estos son step-wise (aplica tanto forward como backward en simultaneo), forward (busca ir agregando variables de manera progresiva) y backward (busca eliminar variables una vez que genera el modelo con todas)
step.wise.result <- stepAIC(model.1, direction = "both", trace = FALSE)
step.wise.result

forward.result <- stepAIC(model.1, direction = "forward", trace = FALSE)
forward.result

backward.result <- stepAIC(model.1, direction = "backward", trace = FALSE)
backward.result

# Se puede observar como en los tres casos las variables seleccionadas por el método de stepAIC son todas las variables utilizadas en el modelo en un principio.

# Dado que los supuestos no se cumplen para poder utilizar el modelo a la hora de predecir, algunas alternativas, tal como se mencionó previamente , son transformar las variables. Otra alternativa es probar con modelo robustos.

robust.model <- rlm(Global_active_power ~ Global_intensity + Sub_metering_3, data = sample.energy)
summary(robust.model)
# Se puede ver como resultado que la nueva ecuación es: -0.0095 + 0.2358 * Global_intensity + 0.0023 * Sub_metering_3.
# En este caso, al ser un modelo robusto, no tiene en cuenta el test de Wald para los distintos coeficientes beta, ni tampoco hay que verificar los supuestos previamente realizados para los modelos anteriores.

predict(robust.model, data.frame(Global_intensity = 0.2, Sub_metering_1 = 0, Sub_metering_3 = 1, Global_active_power = 0.08), interval = "prediction")
# Se puede observar que, a la hora de predecir, el valor obtenido por el modelo robusto es ligeramente mejor al valor obtenido por el modelo que no cumplía con los supuestos.

##### REGRESIÓN LOGISTICA BINARIA ##### 

# Creo train y test split de 70-30 sin repetición.
train.indices <- createDataPartition(sample.energy$avg.consumption, p=.7, list=F, times=1)
train.energy.logit <- sample.energy[train.indices]
test.energy.logit <- sample.energy[-train.indices]

# Realizo la tabla de contingencia entre la variable categórica avg.consumption.device y la variable target con el fin de observar si existe alguna relación entre los valores presentes en ellas.
sjt.xtab(train.energy.logit$avg.consumption, train.energy.logit$avg.consumption.device, show.row.prc = TRUE, show.col.prc = TRUE)
# En el resultado se puede observar cómo existe una relación entre las variables previamente mencionadas, donde un alto avg.consumption.device lleva a un alto avg.consumption de energía. Además, la hipótesis nula de independencia de las variables se rechaza dado el p-value tan bajo obtenido del test de Chi cuadrado, es por esto que la variable resultara útil para introducir en el modelo.

# Genero un gráfico de mosaicos para ver cómo se comportan las variables cualitativas
mosaic(~ avg.consumption + avg.consumption.device, data = train.energy.logit,
       main = "Consumo total vs Consumo por dispositivo", shade = TRUE, legend = TRUE)
# Nuevamente en el grafico se puede observar lo mismo que se desarrolló en la conclusión de la tabla de contingencia previamente realizada.

# Genero el modelo con todas las variables disponibles
model.3 <- glm(avg.consumption ~ Global_intensity + Sub_metering_1 + Sub_metering_2 + Sub_metering_3 + Global_reactive_power + Voltage + avg.consumption.device, data = train.energy.logit, family = "binomial")

summary(model.3)
# Al obtener el summary del modelo se puede observar que hay varias variables en las cuales no se rechaza la hipótesis nula del coeficiente beta. Estas son: Sub_metering_1, Sub_metering_2 y avg.consumption.device, esto quiere decir que podrían llegar a no ser útiles para el modelo en cuestión.

# Para validar esto último se aplicará la selección de las variables predictoras con el fin de poder detectar aquellas que realmente no son útiles. Para eso se aplicará el método step-wise mencionado previamente.
step.wise.result.logit <- stepAIC(model.3, direction = "both", trace = FALSE)
step.wise.result.logit
# Como resultado se obtiene que las únicas variables que son útiles son Global_intensity, Sub_metering_3, Global_reactive_power y Voltage quedando como ecuación de la regresión logística: e^(-55.5297 + 11.7275 * Global_intensity - 0.3674 * Sub_metering_3 - 13.1453 * Global_reactive_power + 0.1098 * Voltage) / (1 + e^(-55.5297 + 11.7275 * Global_intensity - 0.3674 * Sub_metering_3 - 13.1453 * Global_reactive_power + 0.1098 * Voltage))

# Genero el modelo propuesto por step-wise
model.4 <- glm(formula = avg.consumption ~ Global_intensity + Sub_metering_3 + Global_reactive_power + Voltage, family = "binomial", data = train.energy.logit)

# Se obtiene que en este caso todas las variables son útiles para el modelo dado que su valor de beta es significativamente distinto de cero.
summary(model.4)

# Obtengo los coeficientes del modelo mencionados en la ecuación previamente
coefficients(model.4)
# Obtengo los intervalos de confianza que demuestran que el cero no está incluido, lo cual hace que se rechace la hipótesis nula de que beta n = 0
confint(model.4, level=0.95) 

# Valido con el set de testeo
predicted.consumption.category <- predict(model.4, test.energy.logit)
predicted.consumption.category <- ifelse(predicted.consumption.category > 0.5, 1, 0)
table <- table(test.energy.logit$avg.consumption, predicted.consumption.category)
confusionMatrix(table)

# Obtengo el accuracy del modelo. En este caso es un accuracy muy alto el que tiene el modelo. Dependiendo el caso del modelo, esta puede no ser una medida significativa si la cantidad de clases en la variable target se encuentran desbalanceadas. Para el caso de este modelo, donde la cantidad de elementos de la variable target perteneciente a ambas clases se encuentra balanceado es una medida útil, por lo cual se puede concluir que el modelo predice de manera satisfactoria la variable target.
round((sum(diag(table)) / sum(table)) * 100, 2)
summary(test.enery.logit)

# Obtengo los odds ratio (chance) de las variables del modelo
odds.ratio(model.4)
# En este caso se puede observar cómo hay variables "protectoras", como Sub_metering_3 y Global_reactive_power, que en el caso de que su valor aumente van a tender a disminuir la probabilidad de que avg.consumption salga con un uno y otras variables de riesgo como Voltage o Global_intensity que van a tender a aumentar la probabilidad.
# Es importante destacar la gran relación que existe entre Global_intensity y la variable objetivo. Esto se puede observar claramente en el valor de odds que nos devuelve la sentencia previamente ejecutada y su intervalo de confianza que tiende a mantener esta hipótesis dado que se posiciona en valores muy grandes. Es por esto, que tal vez podría ser útil implementar un modelo más simple únicamente entre las dos variables con el fin de poder predecir la segunda.
# Por otro lado, para el caso de Voltage, se puede interpretar que un aumento de una unidad en la variable tiene una probabilidad del 11% más que si la misma no aumentará para obtener un 1 en la variable avg.consumption.

# Calculo la curva ROC del modelo
roc(test.enery.logit$avg.consumption, predicted.consumption.category, print.auc = T, plot =T)
# Claramente se puede observar como la gran importancia de la variable Global_intensity y su alta relación con la variable target dado que el valor de la curva ROC es casi 1 para la muestra aleatoria de datos que se obtuvieron al principio de este script. Esto quiere decir que el modelo predecirá de manera correcta prácticamente la totalidad de los casos que se le presenten.
# Esta situación mencionada previamente es muy poco probable que ocurra en la práctica.

# A fines prácticos se proseguirá a generar el modelo nuevamente, pero sin la variable Global_intensity con el fin de ver que tan bien le va con la curva ROC.
model.5 <- glm(avg.consumption ~ Sub_metering_1 + Sub_metering_2 + Sub_metering_3 + Global_reactive_power + Voltage + avg.consumption.device, data = train.energy.logit, family = "binomial")

step.wise.result.logit <- stepAIC(model.5, direction = "both", trace = FALSE)

model.6 <- glm(formula = avg.consumption ~ Sub_metering_1 + Sub_metering_2 +  Sub_metering_3 + Global_reactive_power + Voltage + avg.consumption.device, family = "binomial", data = train.energy.logit)

predicted.consumption.no.global.intensity <- predict(model.6, test.energy.logit)
predicted.consumption.no.global.intensity <- ifelse(predicted.consumption.no.global.intensity > 0.5, 1, 0)
table.no.global.intensity <- table(test.energy.logit$avg.consumption, predicted.consumption.no.global.intensity)
confusionMatrix(table.no.global.intensity)

round((sum(diag(table.no.global.intensity)) / sum(table.no.global.intensity)) * 100, 2)
# En este caso se puede observar como las métricas obtenidas a partir del modelo no son igual de buenas que en el modelo anterior, pero se nota como las mismas siguen siendo igual de buenas para un modelo de regresión.

roc(test.energy.logit$avg.consumption, predicted.consumption.no.global.intensity, print.auc = T, plot =T)
# Al calcular la curva ROC también se puede notar como el AUC obtenido es alto (0.8747), lo cual nos indica una alta probabilidad de clasificar de manera correcta una observación en base a nuestro modelo.