import QtQuick
import QtQuick.Controls

Rectangle {
    property string text: "HOLD"
    property color tone: text === "PASS" ? "#39D98A" : text === "FAILED" || text === "REJECT" ? "#FF647C" : "#F4B860"
    implicitWidth: label.implicitWidth + 18
    implicitHeight: 24
    radius: 12
    color: Qt.rgba(tone.r, tone.g, tone.b, 0.13)
    border.color: Qt.rgba(tone.r, tone.g, tone.b, 0.55)
    Label { id: label; anchors.centerIn: parent; text: parent.text; color: parent.tone; font.pixelSize: 11; font.weight: Font.DemiBold }
}
