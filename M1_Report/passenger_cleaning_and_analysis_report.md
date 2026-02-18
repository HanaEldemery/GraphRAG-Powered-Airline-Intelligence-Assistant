<h1 align="center">Milestone 1 Report (Passenger Booking csv) - Team 74</h1>


## Passenger Booking Data cleaning Summary


### What we did in the Passenger Booking Dataset
- Standardized column names to a simple, consistent format (lowercase with underscores) so they’re easy to reference.
- Made a quick check of the dataset to spot any issues early.
- Made a check for any missing values
- Reviewed all categorical fields to understand the labels and their frequencies, looking for odd or inconsistent values.
- Removed duplicate rows to keep one version of each record.
- Looked closely at important numeric fields with simple visuals (histograms and boxplots) to see values and potential outliers:
	- num_passengers
	- purchase_lead (days between booking and flight)
	- length_of_stay
	- flight_hour
	- flight_duration

### Outlier detection & handling
To keep the data realistic, we applied practical limits and removed rows outside these ranges:
- num_passengers: 1–6 per booking
- purchase_lead: 0–365 days
- length_of_stay: 0–180 days

Result: extreme, unrealistic records were dropped, improving overall data quality.


### Smoothing very skewed numbers
Some features had a few very large values:
- purchase_lead
- length_of_stay

We applied a log transform to create better distributed versions of these (purchase_lead_log and length_of_stay_log), the transformation clearly improved the distribution of the columns. In simple terms, this compresses very large values onto a gentler scale, which often helps both visuals and predictive models.

#### Log Transformation Effect:
![alt text](..\figs\passanger_booking\log_transformation_effect.png)


### In the end
- Original columns are still there (didn’t drop anything)
- Export of the cleaned dataframe to CSV is available in the notebook  


