# Survey Data Inflight Satisfaction Score

## Overview

This document details the data cleaning process applied to the survey data inflight satisfaction score dataset. The original dataset contained 47,074 rows and 31 columns abd 0 duplicate rows. 4 of these columns (flight_number, arrival_delay_minutes, number_of_legs and actual_flown_miles) were of type int64, and the remaining columns are of type object.

## Data Types Correction

Converted scheduled_departure_date column from an object column into a datetime64 column for proper date handling.

## Handling Missing Values

Initially, the missing values were as follows:

- satisfaction_type -> 12111
- cabin_name -> 19980
- entity -> 3
- loyalty_program_level -> 11616
- departure_gate -> 97
- arrival_gate -> 527
- media_provider -> 1539

### Merging Cabin Information

The cabin_name and cabin_code_desc columns contained overlapping information. After exploring the value distributions, it was observed that cabin_name labelled whether the Economy cabin_code_desc was Economy or Economy Plus. Thus, the Economy values in cabin_code_desc were replaced with their corresponding cabin_name.

### Filling Categorical Missing Values

After looking at the data we found that:

- Missing satisfaction_type values corresponded to rows with a question_text of "What item did you choose?". These missing values meant that the question the passenger was answering was what meal did they choose on the flight and that they didn't answer a satisfaction related question, so we filled these missing values with "not_answered".
- Missing loyalty_program_levels values meant that the passenger was not a member of any loyalty program. Thus, these missing values were filled with "np_program".
- Missing media_provider values meant that the flight that the passenger was on didn't have a media provider. Thus, these missing values were filled with "np_provider".

### Dropping Rows with Missing Values

After investigation, we found that:

- The categorical columns (arrival_gate, departure_gate) logically shouldn't be imputed since this would probably result in these flights having incorrect gates that could be in different airports or countries. Morever, the rows with missing values represented about 1% of the data. Given that the dataset is very large, the logical solution was dropping the rows with these missing values.
- Similarly, since there were only 3 rows with missing entity values, they were dropped as well.

## Handling Unneccessary Columns

After looking at the data we found that:

- The driver_sub_group1 column contained only 1 value "food & beverage" repeated along all rows. This data was redundant so the whole column was dropped.
- The driver_sub_group2 column had 2 values "food and beverage satisfaction" and "comp". The "food and beverage satisfaction" rows were the rows with a value of "How satisfied were you with the food & beverage served on your flight from [CITY] to [CITY]?" in the question_text column and the "comp" rows were the rows with the quesion_text "What item did you choose?". Given that this column was only corresponding to the question_text and not providing important data for our analysis, it was dropped.
- The arrival_delay_group column contained 2 values, "Early & Ontime" representing the rows with a number smaller than or equal to zero in the arrival_delay_minutes column, and "Delayed" corresponding to the rows with a number greater than zero in the arrival_delay_minutes column. Given that the arrival_delay_minutes provided us with richer data (exactly how early or late the flight was), it was deemed more relevant to our analysis, and the arrival_delay_group column was dropped.
- Both columns fleet_usage and ua_wax were corresponding to values from the fleet_type_description column. The "Mainline" values in fleet_usage and the "UA" values in the ua_uax columns corresponded to fleet_type_description values beginning with A or B (for example, "A319-100" or "B787-9"), while the "Express" fleet_usage values and "UAX" values represented rows that had a fleet_type_description starting with C or E (for example, "CRJ-200" or "ERJ-175"). Since these columns were inferred from a more descriptive column, both the fleet_usag column and the ua_uax column were deemed redundant and therefore dropped.

## Data Inconsistency

After investigation, we found that:

- The generation column had only 3 rows with a value "NBK", and those 3 rows had a loyalty_program_level value of "NBK" as well. gGiven that these 3 rows were the only rows with the values of "NBK" in both the generation and loyalty_program_level columns, they were dropped.
- The arrival_gate and departure_gate had some values with inconsistent formats ("-1-" and "\*708"). These values represented about 1% of our data, and again, given that our dataset is large, these rows were dropped.

## Data Analysis

- The arrival_delay_minutes_distribution and arrival_delay_minutes_box_plot graphs (found in figs/survey_data_inflight_satisfaction_score) show that the majority of flights are on-time or early, but with some extreme outliers visible
- The actual_flown_miles_distribution and actual_flown_miles_box_plot graphs (found in figs/survey_data_inflight_satisfaction_score) show that there are more short to medium distance flights than long-haul flights. The long tail extends toward higher mileage values.
- We kept outliers from both of these columns since very long delays or trips should contribute to the passengers' satisfaction levels.
