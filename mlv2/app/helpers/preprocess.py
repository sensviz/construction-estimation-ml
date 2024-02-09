from datetime import datetime

def process(data , date):
    latest = data['productdata'][0]['pricing_history'][0]
    old = data['productdata'][0]['pricing_history'][-1]
    oldprice = old['old_unit_price']
    olddate = old['updated_at_timestamp']
    latestprice = latest['new_unit_price']
    latestdate = latest['updated_at_timestamp']
    pricediff = float(latestprice) -  float(oldprice)
    print(pricediff)
    olddate = datetime.strptime(olddate, '%Y-%m-%d %H:%M:%S')
    latestdate = datetime.strptime(latestdate, '%Y-%m-%d %H:%M:%S')

    # Calculate the difference between the two dates
    datedifference = latestdate - olddate

    # Extract the number of days from the timedelta object
    daysdifference = datedifference.days
    print(daysdifference)
    factorvalue = pricediff/daysdifference
    print(factorvalue)
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    finaldays = date - latestdate
    finaldays = finaldays.days
    pridictedprice = round(finaldays * factorvalue)
    print(pridictedprice)
    return float(pridictedprice)+float(latestprice)
