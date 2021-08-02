library(tidyverse)
library(ggcolors)
source("./hbtfmt/figs/plotutil.R")

E <- 100

simdata <- read.csv("./data/sup_hqnet_q.csv")

plotdata <- simdata %>%
  split(., list(.$q.operant, .$q.other), drop = T) %>%
  lapply(., function(d) {
    dev_mean <- mean(d$test / d$baseline)
    deg_mean <- mean(d$operant.degree / E)
    data.frame(q.operant = unique(d$q.operant),
               q.other = unique(d$q.other),
               dev_mean = dev_mean,
               deg_mean = deg_mean)
}) %>%
  do.call(rbind, .)

# supplementary fig 2(a)
ggplot(plotdata) +
  geom_tile(aes(x = q.operant, y = q.other, fill = dev_mean)) +
  thanatos_light_fill_gradient()
  # mytheme(aspect.ratio = 1)

ggsave("./hbtfmt/figs/supfig-02-a.jpg", dpi = 300, width = 10.0, height = 10.0)

# supplementary fig 2(b)
ggplot(plotdata) +
  geom_tile(aes(x = q.operant, y = q.other, fill = deg_mean)) +
  thanatos_light_fill_gradient()
  # mytheme(aspect.ratio = 1)

ggsave("./hbtfmt/figs/supfig-02-b.jpg", dpi = 300, width = 10.0, height = 10.0)
