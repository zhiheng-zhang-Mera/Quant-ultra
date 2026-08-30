import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

ApplicationWindow {
    id: root
    width: 1440; height: 900; minimumWidth: 1100; minimumHeight: 700
    visible: true
    title: "Quant Ultra · Research Workbench"
    color: "#090D11"

    readonly property var pages: ["Dashboard", "New Research", "Research Flow", "Agents", "Evidence Nexus",
        "Experiment Lab", "Twin Implementation", "Verification Matrix", "Statistical Observatory",
        "Adversarial Lab", "Generalization", "Kernel Telemetry", "Research Memory", "Governance Gate",
        "Reports", "Settings"]
    property int activePage: 0

    header: Rectangle {
        height: 58; color: "#0D1319"; border.color: "#202C36"
        RowLayout {
            anchors.fill: parent; anchors.leftMargin: 20; anchors.rightMargin: 20; spacing: 14
            Rectangle { width: 10; height: 28; radius: 3; color: "#2DE2E6" }
            Label { text: "QUANT ULTRA"; color: "#E8F3F7"; font.pixelSize: 17; font.weight: Font.Bold }
            Label { text: "RESEARCH WORKBENCH"; color: "#6F8493"; font.pixelSize: 12 }
            Item { Layout.fillWidth: true }
            StatusPill { text: workbench.executionMode; tone: workbench.demoMode ? "#F4B860" : "#39D98A" }
            Label { text: workbench.activeRun || "No active run"; color: "#8EA0AF"; font.family: "monospace" }
        }
    }

    RowLayout {
        anchors.fill: parent; spacing: 0
        Rectangle {
            Layout.preferredWidth: 218; Layout.fillHeight: true; color: "#0C1217"; border.color: "#202C36"
            ListView {
                anchors.fill: parent; anchors.margins: 9; spacing: 3; clip: true; model: root.pages
                delegate: ItemDelegate {
                    required property string modelData; required property int index
                    width: ListView.view.width; height: 39
                    text: (index < 9 ? "0" : "") + (index + 1) + "  " + modelData
                    highlighted: root.activePage === index
                    onClicked: root.activePage = index
                    contentItem: Label { text: parent.text; color: parent.highlighted ? "#2DE2E6" : "#93A5B3"; font.pixelSize: 12 }
                    background: Rectangle { radius: 5; color: parent.highlighted ? "#142B32" : "transparent"; border.color: parent.highlighted ? "#245460" : "transparent" }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true; Layout.fillHeight: true; color: "#090D11"
            ColumnLayout {
                anchors.fill: parent; anchors.margins: 22; spacing: 15
                RowLayout {
                    Layout.fillWidth: true
                    Column {
                        Label { text: root.pages[root.activePage]; color: "#E8F3F7"; font.pixelSize: 24; font.weight: Font.DemiBold }
                        Label { text: pageDescription(root.activePage); color: "#718695"; font.pixelSize: 12 }
                    }
                    Item { Layout.fillWidth: true }
                    StatusPill { text: root.activePage === 13 ? "HARD GATE" : "LIVE"; tone: root.activePage === 13 ? "#F4B860" : "#2DE2E6" }
                }

                GridLayout {
                    visible: root.activePage === 0
                    columns: 4; columnSpacing: 12; rowSpacing: 12; Layout.fillWidth: true
                    MetricCard { title: "Active run"; value: workbench.activeRun ? "1" : "0"; detail: "Persistent lifecycle"; Layout.fillWidth: true }
                    MetricCard { title: "Verification"; value: workbench.verificationPassed + " / 12"; detail: "No aggregate score"; Layout.fillWidth: true }
                    MetricCard { title: "Critical holds"; value: workbench.verificationHolds.toString(); detail: "Fail-closed evidence state"; Layout.fillWidth: true }
                    MetricCard { title: "Lifecycle / admission"; value: workbench.lifecycleStatus; detail: workbench.admissionAction; Layout.fillWidth: true }
                }

                Rectangle {
                    visible: root.activePage === 1; Layout.fillWidth: true; Layout.preferredHeight: 190
                    radius: 8; color: "#121920"; border.color: "#263441"
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 18; spacing: 10
                        Label { text: "Research question"; color: "#BAC9D2" }
                        TextArea { id: question; Layout.fillWidth: true; Layout.fillHeight: true; placeholderText: "State a falsifiable quantitative research question…"; color: "#E8F3F7"; wrapMode: TextEdit.Wrap; background: Rectangle { color: "#0B1116"; border.color: question.activeFocus ? "#2DE2E6" : "#2A3945"; radius: 5 } }
                        RowLayout {
                            Button { text: "Create governed run"; enabled: question.text.trim().length > 0; onClicked: workbench.createResearch(question.text) }
                            Button { text: "Create and start"; enabled: question.text.trim().length > 0; onClicked: workbench.startResearch(question.text) }
                            Label { text: "Commands execute outside the GUI thread"; color: "#718695" }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#10171D"; border.color: "#25333E"
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 14
                        Label { text: root.activePage === 2 ? "R0–R20 deterministic lifecycle" : root.activePage === 7 ? "Explicit verification levels" : "Evidence-bound workspace"; color: "#BAC9D2"; font.weight: Font.DemiBold }
                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true; clip: true
                            model: root.activePage === 2 ? workbench.graph : root.activePage === 7 ? workbench.verification : workbench.events
                            delegate: Rectangle {
                                required property var model
                                width: ListView.view.width; height: 44; color: index % 2 ? "#0E151B" : "transparent"
                                RowLayout {
                                    anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10
                                    Label { text: root.activePage === 2 ? model.stageId : root.activePage === 7 ? model.level : model.sequence; color: "#2DE2E6"; font.family: "monospace"; Layout.preferredWidth: 70 }
                                    Label { text: root.activePage === 2 ? model.name : root.activePage === 7 ? "Independent evidence dimension" : model.eventType; color: "#D7E3E9"; Layout.fillWidth: true; elide: Text.ElideRight }
                                    StatusPill { text: root.activePage === 2 || root.activePage === 7 ? model.status : "EVENT"; tone: "#8B7CFF" }
                                }
                            }
                            Label { anchors.centerIn: parent; visible: parent.count === 0; text: "No evidence has been produced for this view."; color: "#607583" }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.preferredWidth: 270; Layout.fillHeight: true; color: "#0C1217"; border.color: "#202C36"
            Column {
                anchors.fill: parent; anchors.margins: 16; spacing: 13
                Label { text: "INSPECTOR"; color: "#6F8493"; font.pixelSize: 11; font.weight: Font.Bold }
                Label { text: root.pages[root.activePage]; color: "#E8F3F7"; font.pixelSize: 16 }
                Rectangle { width: parent.width; height: 1; color: "#24313B" }
                Label { text: "Evidence references"; color: "#9EB0BC" }
                Label { width: parent.width; wrapMode: Text.Wrap; text: "Select an event, verification level, graph node, or decision to inspect immutable provenance here."; color: "#697E8C"; font.pixelSize: 12 }
                StatusPill { text: "NO SELECTION"; tone: "#718695" }
            }
        }
    }

    function pageDescription(index) {
        const descriptions = ["Operational truth without composite scoring", "Preregister a falsifiable research request", "Lifecycle state and critical dependencies", "Roles, providers and information firewalls", "Lineage, origin families and reconciliation", "Frozen specifications and experiment budgets", "Blind N-version comparison from L1 to L6", "V1–V12 remain independently visible", "Multiplicity and dependence-aware inference", "Placebos, attacks and runtime sentinels", "Frozen transfer across required axes", "Kernel runs, environments and fault telemetry", "Searchable successes and failures", "Policy hash, dissent and human authorization", "Reproducible export bundles", "Accessibility, motion and local preferences"]
        return descriptions[index]
    }
}
