library(dplyr)
library(ggplot2)
library(data.table)

path_data_2017 <- '../data/reviews_2017.tsv'
path_data_2018 <- '../data/reviews_2018.tsv'

# 2017
da_2017 <- fread(path_data_2017, sep = '\t')
da_2018 <- fread(path_data_2018, sep = '\t')
da_bind <- da_2017 %>% bind_rows(da_2018)

# statistics
avg_reviews <- da_bind %>% filter(!is.na(stars) & !is.na(text)) %>% 
  select(review_id, business_id, stars, text) %>% 
  group_by(business_id) %>% 
  summarise(Num_review = length(review_id),stars=mean(stars), 
            text=paste(text[1:min(length(text), 40)], collapse = "\\<split\\>")) %>%
  mutate(level=cut(stars, c(0.75, 1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75, 5)))

levels(avg_reviews$level) <- c("1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5")

# distribution of number of reviews for business
avg_reviews %>% as.data.frame() %>%
  ggplot() + geom_histogram(aes(x=Num_review), bins=100)+
  xlim(c(0, 100)) + theme_classic()

# 