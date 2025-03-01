import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='weatherapplicationserver.database.windows.net',
                             user='weatheremergencyapplication',
                             password='Weather123',
                             database='weatherapplication',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection:
        with connection.cursor() as cursor:
            # Read all records for a specific user
            sql = "SELECT `ID`, `Category`, `Location` FROM `Bluesky_Posts` WHERE `OP_user`=%s"
            cursor.execute(sql, ('storm_tracker',))  # Query for a specific user
            results = cursor.fetchall()  # Fetch all matching rows
            for row in results:
                print(row)
finally:
    connection.close()