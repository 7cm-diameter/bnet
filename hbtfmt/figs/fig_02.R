library(tidyverse)
library(ggcolors)
library(GGally)
library(sna)
library(network)
library(gridExtra)
source("./hbtfmt/figs/plotutil.R")

# number of edges on the behavioral network
E <- 100

simdata <- read.csv("./data/sim_01.csv")
matfiles <- list.files("./data", full.names = T, pattern = "sim_01-mat")

# preprocessing for figure 2(a, b)
plotdata_ab <- simdata %>%
  split(., list(.$q.operant), drop = T) %>%
  lapply(., function(d) {
    dev_ratio <- d$test / d$baseline
    dev_mean <- mean(dev_ratio)
    dev_se <- sd(dev_ratio) / sqrt(nrow(d))
    deg_prop <- d$operant.degree / E
    deg_mean <- mean(deg_prop)
    deg_se <- sd(deg_prop) / sqrt(nrow(d))
    data.frame(q = unique(d$q.operant),
               dev_mean = dev_mean, dev_se = dev_se,
               deg_mean = deg_mean, deg_se = deg_se)
}) %>%
  do.call(rbind, .)

# fig 2(a)
ggplot(data = plotdata_ab) +
  geom_line(aes(x = q, y = dev_mean), size = 1.5, color = BLACK) +
  geom_errorbar(aes(x = q,
                    ymax = dev_mean + dev_se,
                    ymin = dev_mean - dev_se),
                width = 0.01, size = 1., color = BLACK) +
  geom_point(aes(x = q, y = dev_mean), size = 7., color = "white") +
  geom_point(aes(x = q, y = dev_mean), size = 5., color = BLACK) +
  ylim(0., 1.) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-02-a.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 2(b)
ggplot(data = plotdata_ab) +
  geom_line(aes(x = q, y = deg_mean), size = 1.5, color = BLACK) +
  geom_errorbar(aes(x = q,
                    ymax = deg_mean + deg_se,
                    ymin = deg_mean - deg_se),
                width = 0.01, size = 1., color = BLACK) +
  geom_point(aes(x = q, y = deg_mean), size = 7., color = "white") +
  geom_point(aes(x = q, y = deg_mean), size = 5., color = BLACK) +
  ylim(0., 0.5) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-02-b.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 2(c)
ggplot(data = simdata) +
  geom_point(aes(x = operant.degree / E, y = test / baseline),
             size = 3, alpha = 0.75, color = BLACK) +
  xlim(0., 0.5) +
  ylim(0., 1.) +
  mytheme(aspect.ratio = 1.)

ggsave("./hbtfmt/figs/fig-02-c.jpg", dpi = 300, width = 10.0, height = 10.0)

# fig 2(d)
show_graph <- function(filename, ...) {
  mat <- read.csv(filename, header = F) %>% data.matrix
  net <- network(mat, directed = F)

  p <- ggnet2(net, edge.size = 0.5, ...) +
    theme(aspect.ratio = 1.,
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.title = element_blank())
  return(p)
}

node_colors <- c(RED, rep(BLACK, 49))

matfiles[c(1, 2, 5, 10)]
networks <- c(1, 2, 5, 10) %>%
  lapply(., function(i) {
    return(show_graph(matfiles[i], color = node_colors, size = 5))
})

netplot <- grid.arrange(networks[[1]], networks[[2]],
                        networks[[3]], networks[[4]])

ggsave("./hbtfmt/figs/fig-02-d.jpg", plot = netplot,
       dpi = 300, width = 10.0, height = 10.0)
