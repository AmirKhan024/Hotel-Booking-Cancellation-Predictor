import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask("__name__")

model = joblib.load('model.sav')

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    hotel = request.form['hotel']
    lead_time = request.form['lead_time']
    is_repeated_guest = request.form['is_repeated_guest']
    previous_cancellations = request.form['previous_cancellations']
    previous_bookings_not_canceled = request.form['previous_bookings_not_canceled']
    days_in_waiting_list = request.form['days_in_waiting_list']
    adr = request.form['adr']
    total_of_special_requests = request.form['total_of_special_requests']
    family_size = request.form['family_size']
    year = request.form['year']
    meal = request.form['meal']
    market_segment = request.form['market_segment']
    distribution_channel = request.form['distribution_channel']
    reserved_room_type = request.form['reserved_room_type']
    deposit_type = request.form['deposit_type']
    customer_type = request.form['customer_type']

    data = [[hotel, lead_time, is_repeated_guest, previous_cancellations, 
             previous_bookings_not_canceled, days_in_waiting_list, adr, 
             total_of_special_requests, family_size, year, meal, market_segment,
             distribution_channel, reserved_room_type, deposit_type, customer_type]]

    new_df = pd.DataFrame(data, columns=['hotel', 'lead_time', 'is_repeated_guest', 'previous_cancellations', 
                                         'previous_bookings_not_canceled', 'days_in_waiting_list', 'adr', 
                                         'total_of_special_requests', 'family_size', 'year', 'meal', 
                                         'market_segment', 'distribution_channel', 'reserved_room_type', 
                                         'deposit_type', 'customer_type'])

    new_df_dummies = pd.get_dummies(new_df[['hotel', 'meal', 'market_segment', 'distribution_channel', 
                                             'reserved_room_type', 'deposit_type', 'customer_type']])

    new_df_dummies = new_df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(new_df_dummies)
    probablity = model.predict_proba(new_df_dummies)[:, 1]

    if prediction == 1:
        result = "The booking is likely to be canceled!"
        confidence = f"Confidence: {probablity[0]*100:.2f}%"
    else:
        result = "The booking is likely to be confirmed!"
        confidence = f"Confidence: {probablity[0]*100:.2f}%"
    
    return render_template('home.html', output1=result, output2=confidence,
                           hotel=hotel, lead_time=lead_time, is_repeated_guest=is_repeated_guest,
                           previous_cancellations=previous_cancellations, 
                           previous_bookings_not_canceled=previous_bookings_not_canceled,
                           days_in_waiting_list=days_in_waiting_list, adr=adr,
                           total_of_special_requests=total_of_special_requests, 
                           family_size=family_size, year=year, meal=meal,
                           market_segment=market_segment, distribution_channel=distribution_channel,
                           reserved_room_type=reserved_room_type, deposit_type=deposit_type,
                           customer_type=customer_type)

if __name__ == '__main__':
    app.run(debug=True)
