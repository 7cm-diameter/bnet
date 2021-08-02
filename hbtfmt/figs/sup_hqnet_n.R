library(tidyverse)
source("./hbtfmt/figs/plotutil.R")

simdata <- read.csv("./data/sup_hqnet_n.csv")

plotdata <- simdata %>%
  split(., list(.$n, .$q.operant)) %>%
  lapply(., function(d) {
    dev_ratio <- d$test / d$baseline
    dev_mean <- mean(dev_ratio)
    dev_se <- sd(dev_ratio) / sqrt(nrow(d))
    deg_prop <- d$operant.degree / (unique(d$n) * 2)
    deg_mean <- mean(deg_prop)
    deg_se <- sd(deg_prop) / sqrt(nrow(d))
    data.frame(n = unique(d$n),
               q = unique(d$q.operant),
               dev_mean = dev_mean, dev_se = dev_se,
               deg_mean = deg_mean, deg_se = deg_se)
}) %>%
  do.call(rbind, .)

# supplementary figure 1(a)
ggplot(data = plotdata) +
  geom_line(aes(x = q, y = dev_mean), size = 1., color = BLACK) +
  geom_errorbar(aes(x = q,
                    ymax = dev_mean + dev_se,
                    ymin = dev_mean - dev_se),
                width = 0.02, size = 1., color = BLACK ) +
  geom_point(aes(x = q, y = dev_mean), size = 5., color = "white") +
  geom_point(aes(x = q, y = dev_mean), size = 3., color = BLACK) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n)

ggsave("./hbtfmt/figs/supfig-01-a.jpg", dpi = 300, width = 10.0, height = 10.0)

# supplementary figure 1(b)
ggplot(data = plotdata) +
  geom_line(aes(x = q, y = deg_mean), size = 1., color = BLACK) +
  geom_errorbar(aes(x = q,
                    ymax = deg_mean + deg_se,
                    ymin = deg_mean - deg_se),
                width = 0.02, size = 1., color = BLACK) +
  geom_point(aes(x = q, y = deg_mean), size = 5., color = "white") +
  geom_point(aes(x = q, y = deg_mean), size = 3., color = BLACK) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n)

ggsave("./hbtfmt/figs/supfig-01-b.jpg", dpi = 300, width = 10.0, height = 10.0)

# supplementary figure 1(c)
ggplot(data = simdata) +
  geom_point(aes(x = operant.degree / (n * 2), y = test / baseline),
             size = 3, alpha = 0.5, color = BLACK) +
  xlim(0., 0.5) +
  ylim(0., 1.) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n, nrow = 1)

ggsave("./hbtfmt/figs/supfig-01-c.jpg", dpi = 300, width = 20.0, height = 5.0)
