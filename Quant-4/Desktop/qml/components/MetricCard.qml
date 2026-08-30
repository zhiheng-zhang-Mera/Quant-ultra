import QtQuick
import QtQuick.Controls

Rectangle {
    property string title: "Metric"
    property string value: "—"
    property string detail: "No evidence"
    implicitHeight: 112
    radius: 8
    color: "#141B22"
    border.color: "#263441"
    Column {
        anchors.fill: parent; anchors.margins: 16; spacing: 7
        Label { text: title; color: "#8EA0AF"; font.pixelSize: 12 }
        Label { text: value; color: "#E8F3F7"; font.pixelSize: 25; font.weight: Font.DemiBold }
        Label { text: detail; color: "#6F8493"; font.pixelSize: 11 }
    }
}
