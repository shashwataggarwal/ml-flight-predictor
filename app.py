from train_clf import load_model, get_X_series

clf = load_model(file_name='mlp_classifier.pkl')

paths = ["Bengaluru-New Delhi", "Mumbai-Bengaluru", "Mumbai-Goa",
         "Mumbai-New Delhi", "New Delhi-Goa", "New Delhi-Kolkata",  "New Delhi-Mumba"]
airlines = ["Air India", "AirAsia", "Go Air", "IndiGo", "Spicejet", "Vistara"]
days_to_depart = 1
times = ["afternoon", "evening", "morning", "night"]

if __name__ == "__main__":
    print("Select flight path:")
    for i, path in enumerate(paths):
        print(i, path)
    path_index = int(input())

    print("Select airline index:")
    for i, airline in enumerate(airlines):
        print(i, airline)
    airline_index = int(input())

    print("Select time of departure:")
    for i, time in enumerate(times):
        print(i, time)
    time_index = int(input())

    print("Select day of departure: (Sunday = 0)")
    day = int(input())

    print("How many days to depart?")
    days_to_depart = int(input())

    print("Enter current fare:")
    fare = float(input())

    airline = airlines[airline_index]
    path = paths[path_index]
    day_time = f"{times[time_index]}-{day}"

    X = get_X_series(airline, days_to_depart, path, day_time)

    print(X)

    y = clf.predict(X)

    print(y)
