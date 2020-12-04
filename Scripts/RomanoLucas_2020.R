

rm(list = ls())
gc()
options(scipen = 999) ### turn off scientific notation


library(data.table)
library(corrplot)
library(psych)
library(PerformanceAnalytics)
library(ggplot2)
library(car)
library(olsrr)

setwd("L:/Data/Projects/regresion-avanzada")

energy <- fread("./Data/household_power_consumption.txt")

summary(energy)

# Transformo las variables numericas correspondientes
energy$Global_active_power <- as.numeric(energy$Global_active_power)
energy$Global_intensity <- as.numeric(energy$Global_intensity)
energy$Global_reactive_power <- as.numeric(energy$Global_reactive_power)
energy$Voltage <- as.numeric(energy$Voltage)
energy$Sub_metering_1 <- as.numeric(energy$Sub_metering_1)
energy$Sub_metering_2 <- as.numeric(energy$Sub_metering_2)
energy$Sub_metering_3 <- as.numeric(energy$Sub_metering_3)

# Tras hacer la transformación aparecen varias filas que se encuentran en NA y parecen ser las mismas para todos los casos. Es por esto que se proseguiran a remover las mismas del dataset
summary(energy)

# Para validar que todos los NA corresponden a los mismos registros se prosigue a eliminar los NA basandome en una unica columna
energy <- na.omit(energy, cols="Sub_metering_3")

# Creo una variable categorica para incluir en el analisis. Esta va a estar relacionada con el consumo promedio de todos los dispositivos y evaluara en "High" o "Low" dependiendo el caso.
energy[, avg.consumption.devices := Sub_metering_1 + Sub_metering_2 + Sub_metering_3]
avg.consupmtion.mean <- mean(energy$avg.consumption.devices)
energy[, avg.consumption := as.factor(ifelse(avg.consumption.devices > avg.consupmtion.mean, "High", "Low"))]
energy[, avg.consumption.devices := NULL]

# Tal como se supuso, todos los NAs observados correspondian a los mismos registros.
summary(energy)

# Creo una particion aleatoria para poder realizar el analisis de los datos con los recursos computacionales disponibles
set.seed(333)
sample.size <- floor(0.1 * nrow(energy))
sample.energy.idx <- sample(seq_len(nrow(energy)), size = sample.size)
sample.energy <- energy[sample.energy.idx]

#### REGRESION LINEAL SIMPLE

# Obtengo las variables numericas para poder evaluar la correlacion
numeric.idx <- which(sapply(sample.energy, class) %in% c("numeric", "integer"))
numeric.sample.energy <- subset(sample.energy, select = numeric.idx)

# Obtengo la correlacion a traves de Spearman y Pearson

lineal.energy.correlation <- cor(numeric.sample.energy)
spearman.energy.correlation <- cor(numeric.sample.energy, y = NULL, use = "everything", method = c("spearman"))

# A fines practicos, dado el alto costo en tiempo de kendall y que el mismo esta diseñado para evaluaar variables ordinales se tomara muestra aleatoria mucho mas pequeña para poder obtener la correlacion
small.sample.energy.idx <- sample(seq_len(nrow(energy)), size = 50)
small.sample.energy <- energy[small.sample.energy.idx]
small.numeric.idx <- which(sapply(sample.energy, class) %in% c("numeric", "integer"))
numeric.small.sample.energy <- subset(sample.energy, select = small.numeric.idx)
kendall.energy.correlation <- cor(numeric.small.sample.energy, y = NULL, use = "everything", method = c("kendall"))
# TODO: Dejarlo ejecutando el tiempo que sea necesario para análisis


corPlot(numeric.sample.energy, cex = 1.1, main = "Matriz de correlacion")
chart.Correlation(numeric.sample.energy, histogram = TRUE, lm = TRUE, method = "pearson")

# Si me basara en la correlacion lineal se podria observar una correlacion muy fuerte entre Global_active_power (variable target) y Global_intensity por lo cual puede llegar a ser util realizar una regresión lineal (en un principio) para poder predecir la variable target. 
# Por otro lado se observan otras correlaciones lineales no tan fuertes, pero que valen la pena analizar, que son la que se da por Sub_metering_3 con Global_active_power o de Sub_metering_1 con Global_active_power, aunque esta ultima es casi despreciable en comparacion a las anteriores.
# En este caso, para poder aplicar Spearman es necesario que las variables sean normales, lo cual no se da salvo en la variable Voltage, esto se observa en el grafico obtenido en la sentencia de chart.Correlation(numeric.sample.energy, histogram = TRUE, lm = TRUE, method = "pearson").
#Es por esto que se opto por aplicar tambien Pearson con el fin de validar la correlacion entre las variables, lo cual termino dando bastante similar a Spearman, con ligeras diferencias en los valores de correlacion obtenidos.


# Grafico de dispersion entre Global_intensity y Global_active_power
ggplot(sample.energy, aes(x=Global_intensity, y=Global_active_power)) +
    geom_point() +
    geom_smooth(method=lm, color="red", fill="#69b3a2", se=T)
# En este grafico se puede ver como claramente las variables Global_intensity y Global_active_power siguen una relacion lineal casi perfecta, donde todos los puntos se ubican cercanos a una funcion lineal que surgira como resultado de entrenar un modelo de regresion lineal sobre los datos.

# Grafico de dispersion entre Sub_metering_3 y Global_active_power
ggplot(sample.energy, aes(x=Sub_metering_3, y=Global_active_power)) +
    geom_point() +
    geom_smooth(method=lm, color="red", fill="#69b3a2", se=T)
# En este caso se puede ver como los valores minimos de Global_active_power se encuentran sobre una recta, pero en general no parece seguir una correlacion lineal. Es importante destacar que en el analisis previo de correlaciones, esta dio un valor muy bajo en comparacion a la anterior comparación.

# Grafico de dispersion entre Sub_metering_1 y Global_active_power
ggplot(sample.energy, aes(x=Sub_metering_1, y=Global_active_power)) +
    geom_point() +
    geom_smooth(method=lm, color="red", fill="#69b3a2", se=T)
# En este caso, al igual que en el anterior se observa el mismo comportamiento en cuanto a los valores minimos, pero a medida que los valores de Global_active_power van aumentando la forma de la relacion escapa la forma lineal deseada para poder aplicar una regresion lineal simple.

# Combinamos todos los graficos realizados previamente en uno solo, además incluyendo la combinación entre las variables independientes
pairs(~Global_active_power + Global_intensity + Sub_metering_3 + Sub_metering_1, data=sample.energy, main="Energy consumption scatterplot")

# Genero la regresion lineal en base a la variable Global_intensity
model.1 <- lm(Global_active_power ~ Global_intensity, data = sample.energy )
summary(model.1)
# Haciendo el summary se puede observar la seguridad con la que se rechaza la hipotesis de que el valor es cero tanto el intercept como el Global_intensity al ver el p-value. Esto era de esperarse tras los analisis de correlacion que se llevaron a cabo, donde se veia la alta correlacion lineal que presentaba esta variable con la de Global_active_power.
# La ecuación de regresión obtenida fue: -0.0082203 + 0.2376743 * Global_intensity
# Cabe destacar que el p-value general dio extremadamente bajo, lo cual era de esperarse ya que evalua si al menos uno de los coeficientes de beta da distinto de cero, que en este caso se cumplia en ambos.

# A fines practicos se realizara la prueba utilizando Sub_metering_1 y Sub_metering_3 en conjunto con Global_active_power dado que tenia una correlacion baja comparada con Global_intensity. En la practica, el modelo optimo seria el generado por Global_intensity por el momento (aun falta validar los supuestos)
model.2 <- lm(Global_active_power ~ Sub_metering_1, data = sample.energy)
summary(model.2)
# En este caso, si bien no rechaza el intercept y la variable Sub_metering_1, el valor de R cuadrado ajustado es mucho mas bajo que el visto en el modelo 1, lo cual hace notar que dicho modelo no seria util para poder predecir el valor de Global_active_power comno lo es el generado por Global_intensity.
# La ecuación de regresión obtenida fue: 0.999286 + 0.082853 * Sub_metering_1


model.3 <- lm(Global_active_power ~ Sub_metering_3, data = sample.energy)
summary(model.3)
# En este caso tambien se rechazan las hipotesis nulas del intercept y de la variable, pero al igual que en el anterior, el R cuadrado ajustado tiene un valor muy bajo en comparacion a lo visto en el primer modelo, por lo cual no seria util dicho modelo.
# La ecuación de regresión obtenida fue: 0.5745365 + 0.0800555 * Sub_metering_3


# TODO: Revisar la clase
par(mfrow = c(2,2))
plot(model.1)
# Al observar los graficos obtenidos se puede notar como:
# 1. # TODO: verlo con valeria
# 2. Los residuos no se distribuyen de manera normal. Se puede ver como al principio, los valores que se encuentran desde -4 a -2 rompen con la normalidad de los errores.
# 3. En el tercer grafico (Scale-Location) se puede observar como los errores siguen una forma random, por lo cual lleva a pensar que se cumple la homocedasticidad de los mismos.
# 4. En el cuarto grafico se observa la influencia de los residuos en la regresion. En este caso se puede observar como los valores que se encuentran más alejados de x = 0 generan un mayor impacot en el modelo

plot(model.2)
# El caso del modelo dos es muy similar a lo que se observa en el modelo uno, a diferencia de los siguientes detalles:
# 1. # TODO
# 2. Los residuos no se comportan de manera normal en ningun momento a diferencia del modelo uno, donde los valores superiores a -2 tenian una forma normal.
# 3. Los valores se comportan de manera random al igual que en el modelo uno, lo cual es bueno al pensar en la homocedasticidad del modelo.
# 4. En este caso, a diferencia del modelo uno, los valores mas influyentes se encuentran alejados de y = 0, pero cercanos a x = 0

plot(model.3)
# En este ultimo modelo, los graficos observados son un tanto distintos.
# 1. # TODO
# 2. A diferencia del modelo uno, los valores siguen la recta normal hasta un valor de uno aproximadamente y despues de alli comienzan a comportarse de manera no normal.
# 3. Los residuos tambien presentan una forma random, como en todos los modelos anteriores
# 4. Los valores influyentes se encuentran alejados de y = 0.

# Obtengo los valores predichos por cada uno de los modelos
predicted.model.1 <- fitted(model.1)
predicted.model.2 <- fitted(model.2)
predicted.model.3 <- fitted(model.3)

# Obtengo los residuos de las predicciones realizadas
residuals.model.1 <- residuals(model.1, type="response")
residuals.model.2 <- residuals(model.2, type="response")
residuals.model.3 <- residuals(model.3, type = "response")

# Unifico todos los valores predichos y los residuos en la misma tabla.
energy.final <- cbind(sample.energy, predicted.model.1, residuals.model.1, predicted.model.2, residuals.model.2, predicted.model.3, residuals.model.3)
# TODO: Analizar valores obtenidos

coefficients(model.1)
coefficients(model.2)
coefficients(model.3)
# Al ejecutar las sentencias de coefficients se obtienen nuevamente los valores de beta cero y beta uno que previamente fueron mencionados al realizar el summary de cada uno de los modelos.

confint(model.1, level=0.95)
confint(model.2, level=0.95)
confint(model.3, level=0.95)
# Al obtener el intervalo de confianza de 0.95 para cada uno de los beta de los distintos modelos, se puede observar como ninguno de los coeficientes tiene en su intervalo el valor cero. Es por esta razon que al generar los modelos se rechaza la hipotesis nula de que el beta n analizado es igual a cero para cada uno de los coeficientes obtenidos (p-value muy bajo visto en el summary del modelo)

# TODO
coef_lmbeta1 <- lm.beta(model.1)

l1=influence(model.1)
l2=influence(model.2)
l2=influence(model.3)

c1=cooks.distance(model.1)
plot(c1,ylab="Distancia de Cook")

c2=cooks.distance(model.2)
plot(c2,ylab="Distancia de Cook")

c3=cooks.distance(model.3)
plot(c3,ylab="Distancia de Cook")

ols_plot_cooksd_bar(model.1)
ols_plot_cooksd_chart(model.1)

ols_plot_cooksd_bar(model.2)
ols_plot_cooksd_chart(model.2)

ols_plot_cooksd_bar(model.3)
ols_plot_cooksd_chart(model.3)

# Residual studentizado
# Estandariza y le agrega un componente de apalancamiento que lo convierte en studentizado
r = rstudent(model.1)
plot(r,ylab="R studentizado")

#leverage (apalancamiento)
ols_plot_resid_lev(model.1)

#####SUPUESTOS#################################

# Shapiro wilks (Normalidad de los errores)
# Genero una muestra aleatoria de los residuos para poder ejecutar shapiro.test ya que el limite en cantidad de variables es de 5000.
set.seed(333)
residuals.model.1.sample <- sample(seq_len(length(residuals.model.1)), size = 5000)
residuals.model.2.sample <- sample(seq_len(length(residuals.model.2)), size = 5000)
residuals.model.3.sample <- sample(seq_len(length(residuals.model.3)), size = 5000)

shapiro.test(residuals.model.1.sample) 
# Al aplicar el test de shapiro para evaluar la normalidad de los errores en el modelo 1 se puede observar como se rechaza la hipotesis de que la muestra aleatoria de los errores sigue una distribucion normal. 
# Por este motivo, el modelo no podrá utilizarse para poder realizar estimaciones. Una alternativa para poder solucionar el problema de la normalidad de los residuos sería aplicar una transformación de Box y Cox.

shapiro.test(residuals.model.2.sample)
shapiro.test(residuals.model.3.sample)
# Tanto para el modelo 2 como para el tres ocurre la misma situacion que para el modelo uno (se rechaza la normalidad de los residuos).

# Estadisticas de los residuos (compruebo media = 0)
summary(residuals.model.1)
summary(residuals.model.2)
summary(residuals.model.3)
# En este caso, al realziar el summary de los distintos residuos se puede observar como, si bien la media es igual a cero, la mediana no lo es, por lo cual no presentan una estructura simetrica.
# A pesar de no ser igual a cero, en el caso del modelo uno se encuentra muy proximo a dicho valor, a diferencia de los otros casos que se encuentran alejados del cero.

# Graficos de dispersion Residual vs Predichos
# En los siguientes graficos se va a poder observar como se relacionan los distintos valores predichos con los residuos. De esta forma se podra observar si existe homocedasticidad de los residuos. La intepretacion de los graficos es similar a la que se realizo en la parte de plot() de los distintos modelos, donde además de este grafico de dispersion se encontraba el grafico QQ-plot de los residuos, como así tambien los graficos de la distancia de Cook.
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

par(mfrow=c(1,1))
attach(sample.energy)
plot(predicted.model.3, residuals.model.3,
     las=2,
     col="blue",
     pch=16,
     xlim=c(-1,10),
     ylim=c(-1,3),
     main="Relacion entre residuales y valores predichos - Modelo 3")
title(ylab="", line=3.3)

# Test de autocorrelacion de los Errores
# Mediante el test de durbin watson voy a validar la correlacion entre los residuos del modelo para poder ver si hay algun tipo de relacion entre los mismos.
set.seed(333)
durbinWatsonTest(model.1)
# Al aplicar durbin watson para los residuos del modelo uno da como resultado que los mismos no se encuentran correlacionados (p-value es muy superior a 0.005).

durbinWatsonTest(model.2)
# Para el caso del modelo dos, si bien tambien da que no se rechaza la hipotesis nula de no correlacion, el p-value es mucho menor que el del modelo 1 y que el que se vera del modelo 3.

durbinWatsonTest(model.3)
# Al igual que para los anteriores, no se rechaza la hipotesis nula de correlacion

# criterios de informacion. Mejor modelo es el que tiene menor AIC
# Sirve para comparar modelos 
# TODO
AIC(model.1, model.2)

# Predecir un valor determinado
# TODO

predict(model.1, data.frame(Global_active_power = 41.8), interval = "prediction")
predict(model.2, data.frame(Global_active_power = 41.8), interval = "prediction")
predict(model.3, data.frame(Global_active_power = 41.8), interval = "prediction")