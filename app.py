
from flask import Flask, render_template
from src.database import DatabaseManager

app = Flask(__name__)


@app.route("/")
def dashboard():

    db = DatabaseManager()

    # Get all detection records
    records = db.get_all_detections()

    # Get separate statistics
    stats = db.get_statistics()

    # Prepare chronological chart data
    chronological_records = list(reversed(records))

    chart_labels = [
        record[1] for record in chronological_records
    ]

    chart_values = [
        record[3] for record in chronological_records
    ]

    # Prepare category counts for comparison chart
    category_labels = ["With Mask", "Without Mask"]

    category_values = [
        stats["mask_detections"],
        stats["no_mask_detections"]
    ]

    return render_template(
        "index.html",

        records=records,

        stats=stats,

        chart_labels=chart_labels,

        chart_values=chart_values,

        category_labels=category_labels,

        category_values=category_values
    )


if __name__ == "__main__":

    app.run(debug=True)
