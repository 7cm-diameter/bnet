library(tidyverse)
source("./hbtfmt/figs/plotutil.R")

E <- 2

simdata_vi <- read.csv("./data/sup_td_variable_interval_n.csv")
simdata_vr <- read.csv("./data/sup_td_variable_ratio_n.csv")
simdata_choice <- read.csv("./data/sup_td_choice_n.csv")
simdata_noncontigent <- read.csv("./data/sup_td_noncontingent_n.csv")

# preprocessing for figure 3(a, b)
preprocess_3ab <- function(d) {
  dev_ratio <- d$test / d$baseline
  dev_mean <- mean(dev_ratio)
  dev_se <- sd(dev_ratio) / sqrt(nrow(d))
  deg_prop <- d$operant.degree / (E * unique(d$n))
  deg_mean <- mean(deg_prop)
  deg_se <- sd(deg_prop) / sqrt(nrow(d))
  data.frame(n = unique(d$n),
             amount.training = unique(d$amount.training),
             dev_mean = dev_mean, dev_se = dev_se,
             deg_mean = deg_mean, deg_se = deg_se)
}

plotdata_3ab_vi <- simdata_vi %>%
  split(., list(.$amount.training, .$n)) %>%
  lapply(., preprocess_3ab) %>%
  do.call(rbind, .)

plotdata_3ab_vr <- simdata_vr %>%
  split(., list(.$amount.training, .$n)) %>%
  lapply(., preprocess_3ab) %>%
  do.call(rbind, .)

# preprocessing for figure 3d
plotdata_3d <- (function() {
                  simdata_choice$cond <- "choice"
                  simdata_noncontigent$cond <- "no choice"
                  rbind(simdata_choice, simdata_noncontigent)
             })()

# supplementary fig 5(a)
ggplot() +
  geom_line(data = plotdata_3ab_vi,
            aes(x = amount.training, y = dev_mean),
            color = RED, size = 1.) +
  geom_line(data = plotdata_3ab_vr,
            aes(x = amount.training, y = dev_mean),
            color = BLACK, size = 1.) +
  geom_errorbar(data = plotdata_3ab_vi,
                aes(x = amount.training,
                    ymax = dev_mean + dev_se,
                    ymin = dev_mean - dev_se),
                color = RED, width = 5, size = 1.) +
  geom_errorbar(data = plotdata_3ab_vr,
                aes(x = amount.training,
                    ymax = dev_mean + dev_se,
                    ymin = dev_mean - dev_se),
                color = BLACK, width = 5, size = 1.) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = dev_mean),
            color = "white", size = 5) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = dev_mean),
            color = "white", size = 5) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = dev_mean),
            color = RED, size = 3) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = dev_mean),
            color = BLACK, size = 3) +
  xlim(20, 205) +
  ylim(0, 1) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n)

ggsave("./hbtfmt/figs/supfig-03-a.jpg", dpi = 300, width = 10.0, height = 10.0)

# supplementary fig 5(b)
ggplot() +
  geom_line(data = plotdata_3ab_vi,
            aes(x = amount.training, y = deg_mean),
            color = RED, size = 1) +
  geom_line(data = plotdata_3ab_vr,
            aes(x = amount.training, y = deg_mean),
            color = BLACK, size = 1) +
  geom_errorbar(data = plotdata_3ab_vi,
                aes(x = amount.training,
                    ymax = deg_mean + deg_se,
                    ymin = deg_mean - deg_se),
                color = RED, width = 5, size = 1) +
  geom_errorbar(data = plotdata_3ab_vr,
                aes(x = amount.training,
                    ymax = deg_mean + deg_se,
                    ymin = deg_mean - deg_se),
                color = BLACK, width = 5, size = 1) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = deg_mean),
            color = "white", size = 5) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = deg_mean),
            color = "white", size = 5) +
  geom_point(data = plotdata_3ab_vi,
            aes(x = amount.training, y = deg_mean),
            color = RED, size = 3) +
  geom_point(data = plotdata_3ab_vr,
            aes(x = amount.training, y = deg_mean),
            color = BLACK, size = 3) +
  xlim(20, 205) +
  ylim(0, 0.5) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n)

ggsave("./hbtfmt/figs/supfig-03-b.jpg", dpi = 300, width = 10.0, height = 10.0)

# supplementary fig 5(c)
ggplot() +
  geom_point(data = simdata_vi,
             aes(x = operant.degree / (E * n), y = test / baseline),
             color = RED, size = 3, alpha = 0.75) +
  geom_point(data = simdata_vr,
             aes(x = operant.degree / (E * n), y = test / baseline),
             color = BLACK, size = 3, alpha = 0.75) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n)

ggsave("./hbtfmt/figs/supfig-03-c.jpg", dpi = 300, width = 10.0, height = 10.0)

# supplementary fig 5(d)
ggplot() +
  geom_boxplot(data = plotdata_3d,
               aes(x = cond, y = test / baseline),
               size = 0.75, color = BLACK) +
  mytheme(aspect.ratio = 1) +
  facet_wrap(~n)

ggsave("./hbtfmt/figs/supfig-03-d.jpg", dpi = 300, width = 10.0, height = 10.0)
