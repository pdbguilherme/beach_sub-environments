panel.signif <-  function (x, y, corr = NULL, col.regions, digits = 2, cex.cor, 
                           ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  results <- cor.test(x, y, alternative = "two.sided")
  est <- results$p.value
  stars <- ifelse(est < 5e-4, "***", 
                  ifelse(est < 5e-3, "**", 
                         ifelse(est < 5e-2, "*", "NS")))
  cex.cor <- 0.4/strwidth(stars)
  text(0.5, 0.5, stars, cex = cex.cor)
}
