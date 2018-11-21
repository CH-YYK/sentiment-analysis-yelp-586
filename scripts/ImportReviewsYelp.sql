## 1. change default mysql local dir to specified dir 
## 2. remove secure-file-priv
## 3. give sql permission to read/write date from/to a dir


USE YelpReviews;

DROP TABLE if exists Yelp_business;

CREATE TABLE Yelp_business (
	`business_id` varchar(22) NOT NULL,
	`stars` float(2,1) default 0.0 not null,
	`name` LONGTEXT default null,
	`categories` mediumtext,
	`latitude` float(5,2) default 0.0 NOT NULL,
	`longitude` float(5,2) default 0.0 NOT NULL,
	`state` varchar(3) default null,
	`review_count` int(10) default 0 NOT NULL,
	primary key (business_id)
);

LOAD DATA INFILE '/home/yikang/Desktop/TextData/business.tsv' 
INTO TABLE Yelp_business 	
LINES TERMINATED BY '\n' 	# important for Text data
ignore 1 rows;

## 

DROP TABLE if exists Yelp_reviews;

CREATE TABLE Yelp_reviews (
    `business_id` varchar(22) NOT NULL,
    
    `date` varchar(10),
    `review_id` varchar(22) NOT NULL,
    `stars` varchar(1) NOT NULL,
    
    `user_id` varchar(22) NOT NULL,
    primary key (review_id),
    foreign key (business_id) references Yelp_business(business_id)
    # `text` LONGTEXT
);

# for year in 2013-2017
LOAD DATA INFILE '/home/yikang/Desktop/TextData/all.tsv' 
INTO TABLE Yelp_reviews 	
LINES TERMINATED BY '\r\n' 	# important for Text data
ignore 1 rows;



## join reviews to business
create table temp as
select Yelp_reviews.stars as review_stars, 
Yelp_business.business_id as Business_id, 
Yelp_business.stars as stars 
from Yelp_business
left join Yelp_reviews on Yelp_business.business_id=Yelp_reviews.business_id;

select Yelp_business.business_id as business_id
from Yelp_business
left join Yelp_reviews on Yelp_business.business_id = yelp
## analyse how estimated stars are associated with real stars.

## group by business_id and save as new
create table temp as 
select AVG(stars) as estimated_stars,
business_id
from Yelp_reviews
group by business_id


