import QtQuick 6.8
import QtQuick3D 6.8

// The Python side supplies normalized scene records.  The normalization is
// uniform, so the specimen-plane frame and every oriented box remain exact
// relative to one another.
Item {
    id: root
    objectName: "lamellarSceneRoot"

    property var slabData: []
    property int selectedBranch: -2147483648
    property string language: "en"
    property real axisLength: 1.2
    property bool publicationActive: false
    property bool publicationMode: false
    property bool showOverlays: true
    property bool transparentBackground: false
    property real publicationRoughness: 0.72
    property bool publicationAO: true
    property string publicationQuality: "publication"
    property var publicationGroups: []

    property real cameraYaw: -38
    property real cameraPitch: -26
    property real cameraDistance: 12
    property real orthoHeight: 8.8
    // Qt Quick 3D's orthographic magnification is a viewport scale.  Keeping
    // a 480-unit reference means a larger export framebuffer gets the same
    // composition at genuinely higher resolution instead of retaining an
    // 80-pixel object in a 2400-pixel image.
    property real cameraReferenceHeight: 480
    property real targetX: 0
    property real targetY: 0
    property real targetZ: 0

    signal cameraChanged(var state)

    function cameraState() {
        return {
            "yaw": cameraYaw,
            "pitch": cameraPitch,
            "distance": cameraDistance,
            "ortho_height": orthoHeight,
            "target": { "x": targetX, "y": targetY, "z": targetZ }
        }
    }

    function updateCamera() {
        cameraRig.position = Qt.vector3d(targetX, targetY, targetZ)
        cameraRig.eulerRotation = Qt.vector3d(cameraPitch, cameraYaw, 0)
        var safeHeight = Math.max(0.1, orthoHeight)
        // Keep one world unit at the same number of pixels in both axes.
        // OrthographicCamera already knows the viewport aspect; multiplying
        // only horizontalMagnification by width/height would stretch every
        // square slab on landscape views and squash it on portrait views.
        var viewportScale = Math.min(Math.max(1, width), Math.max(1, height)) / cameraReferenceHeight
        var magnification = safeHeight * viewportScale
        camera.verticalMagnification = magnification
        camera.horizontalMagnification = magnification
        camera.position = Qt.vector3d(0, 0, Math.max(0.2, cameraDistance))
    }

    function emitCameraChanged() {
        cameraChanged(cameraState())
    }

    onCameraYawChanged: { updateCamera(); emitCameraChanged() }
    onCameraPitchChanged: { updateCamera(); emitCameraChanged() }
    onCameraDistanceChanged: { updateCamera(); emitCameraChanged() }
    onOrthoHeightChanged: { updateCamera(); emitCameraChanged() }
    onTargetXChanged: { updateCamera(); emitCameraChanged() }
    onTargetYChanged: { updateCamera(); emitCameraChanged() }
    onTargetZChanged: { updateCamera(); emitCameraChanged() }
    onWidthChanged: updateCamera()
    onHeightChanged: updateCamera()

    View3D {
        id: view3d
        anchors.fill: parent
        renderMode: root.publicationMode ? View3D.Offscreen : View3D.Inline
        camera: camera
        environment: SceneEnvironment {
            backgroundMode: root.publicationMode && root.transparentBackground
                             ? SceneEnvironment.Transparent : SceneEnvironment.Color
            clearColor: (root.publicationActive || root.publicationMode) && !root.transparentBackground
                         ? "#ffffff" : "#f4f1ea"
            antialiasingMode: root.publicationMode && root.publicationQuality === "publication"
                              ? SceneEnvironment.SSAA : SceneEnvironment.MSAA
            antialiasingQuality: root.publicationMode && root.publicationQuality === "publication"
                                 ? SceneEnvironment.VeryHigh : SceneEnvironment.High
            aoEnabled: root.publicationAO
            aoStrength: root.publicationAO ? 16 : 0
            aoDistance: root.publicationAO ? 1.4 : 0
            aoSoftness: root.publicationAO ? 28 : 50
            aoSampleRate: root.publicationAO ? 3 : 2
            depthPrePassEnabled: root.publicationAO
        }

        Node {
            id: sceneRoot

            DirectionalLight {
                eulerRotation: Qt.vector3d(-35, -35, -20)
                brightness: 1.15
                color: "#ffffff"
                ambientColor: "#5e6b6e"
                castsShadow: root.publicationMode
                shadowMapQuality: Light.ShadowMapQualityHigh
                softShadowQuality: Light.PCF16
                shadowFactor: 38
                shadowMapFar: 40
                shadowBias: 0.02
            }
            DirectionalLight {
                eulerRotation: Qt.vector3d(35, 145, 10)
                brightness: 0.65
                color: "#ffffff"
                ambientColor: "#60696b"
                castsShadow: false
            }
            DirectionalLight {
                eulerRotation: Qt.vector3d(-10, 70, 25)
                brightness: 0.25
                color: "#fffaf2"
                castsShadow: false
            }

            Node {
                id: cameraRig
                OrthographicCamera {
                    id: camera
                    clipNear: 0.01
                    clipFar: 1000
                    position: Qt.vector3d(0, 0, 12)
                }
            }

            // Each slab is a unit cube scaled in its local width/depth/
            // thickness axes.  The quaternion is derived from the supplied
            // column-basis orientation matrix by the Python adapter.
            Repeater3D {
                id: slabs
                model: root.slabData
                visible: !root.publicationActive
                delegate: Model {
                    source: "#Cube"
                    position: Qt.vector3d(modelData.x, modelData.y, modelData.z)
                    // Qt Quick 3D's built-in #Cube primitive spans 100 scene
                    // units, so a 0.01 factor maps its scale to one world
                    // unit and keeps the supplied dimensions exact.
                    scale: Qt.vector3d(modelData.sx * 0.01, modelData.sy * 0.01, modelData.sz * 0.01)
                    rotation: Qt.quaternion(modelData.qw, modelData.qx, modelData.qy, modelData.qz)
                    materials: DefaultMaterial {
                        diffuseColor: Qt.rgba(modelData.r, modelData.g, modelData.b, 1.0)
                        opacity: root.selectedBranch === -2147483648 || modelData.branch === root.selectedBranch
                                 ? modelData.a : modelData.a * 0.22
                        specularAmount: 0.16
                    }
                }
            }

            // Publication geometry is supplied as one merged flat-shaded
            // QQuick3DGeometry per branch/material group.  It consumes the
            // mesh contract directly, so no second box approximation is
            // inferred in QML.
            Repeater3D {
                id: publicationSlabs
                model: root.publicationGroups
                visible: root.publicationActive
                delegate: Model {
                    geometry: modelData.geometry
                    opacity: root.selectedBranch === -2147483648 || modelData.branch === root.selectedBranch
                             ? 1.0 : 0.22
                    materials: PrincipledMaterial {
                        baseColor: Qt.rgba(1, 1, 1, 1)
                        vertexColorsEnabled: true
                        metalness: 0.0
                        roughness: root.publicationRoughness
                        specularAmount: 0.18
                        opacity: 1.0
                    }
                }
            }

            // A restrained specimen-frame helper.  +y is the draw axis and
            // +z points out of the specimen plane, matching the public scene
            // contract.
            Node {
                id: axes
                visible: root.slabData.length > 0 && root.showOverlays && !root.publicationMode
                property real thickness: Math.max(0.018, root.axisLength * 0.018)

                Model {
                    source: "#Cube"
                    position: Qt.vector3d(root.axisLength * 0.5, 0, 0)
                    scale: Qt.vector3d(root.axisLength * 0.01, axes.thickness * 0.01, axes.thickness * 0.01)
                    materials: DefaultMaterial { lighting: DefaultMaterial.NoLighting; diffuseColor: "#bf5b55" }
                }
                Model {
                    source: "#Cube"
                    position: Qt.vector3d(0, root.axisLength * 0.5, 0)
                    scale: Qt.vector3d(axes.thickness * 0.01, root.axisLength * 0.01, axes.thickness * 0.01)
                    materials: DefaultMaterial { lighting: DefaultMaterial.NoLighting; diffuseColor: "#4c9872" }
                }
                Model {
                    source: "#Cube"
                    position: Qt.vector3d(0, 0, root.axisLength * 0.5)
                    scale: Qt.vector3d(axes.thickness * 0.01, axes.thickness * 0.01, root.axisLength * 0.01)
                    materials: DefaultMaterial { lighting: DefaultMaterial.NoLighting; diffuseColor: "#477cc2" }
                }
                Model {
                    source: "#Sphere"
                    scale: Qt.vector3d(axes.thickness * 3.2 * 0.01, axes.thickness * 3.2 * 0.01, axes.thickness * 3.2 * 0.01)
                    materials: DefaultMaterial { lighting: DefaultMaterial.NoLighting; diffuseColor: "#26364b" }
                }
            }
        }
    }

    Rectangle {
        id: legend
        visible: root.showOverlays && !root.publicationMode
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 14
        width: language === "zh" ? 132 : 152
        height: language === "zh" ? 82 : 76
        radius: 9
        color: "#fdfbf6"
        opacity: 0.92
        border.color: "#d8d0c7"
        border.width: 1

        Column {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 3
            Text { text: language === "zh" ? "试样坐标" : "Specimen axes"; color: "#24364b"; font.bold: true; font.pixelSize: 12 }
            Text { text: "X  " + (language === "zh" ? "面内" : "in plane"); color: "#bf5b55"; font.pixelSize: 11 }
            Text { text: "Y  " + (language === "zh" ? "绘制参考" : "draw reference"); color: "#4c9872"; font.pixelSize: 11 }
            Text { text: "Z  " + (language === "zh" ? "面外" : "out of plane"); color: "#477cc2"; font.pixelSize: 11 }
        }
    }

    Rectangle {
        visible: root.showOverlays && !root.publicationMode
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 14
        width: language === "zh" ? 142 : 174
        height: 29
        radius: 8
        color: "#fdfbf6"
        opacity: 0.86
        border.color: "#d8d0c7"
        border.width: 1
        Text {
            anchors.centerIn: parent
            text: language === "zh" ? "拖动旋转 · 滚轮缩放" : "Drag to orbit · wheel to zoom"
            color: "#526174"
            font.pixelSize: 10
        }
    }

    MouseArea {
        id: interaction
        anchors.fill: parent
        visible: root.showOverlays && !root.publicationMode
        hoverEnabled: false
        property bool dragging: false
        property real lastX: 0
        property real lastY: 0

        onPressed: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                dragging = true
                lastX = mouse.x
                lastY = mouse.y
                mouse.accepted = true
            }
        }
        onPositionChanged: function(mouse) {
            if (!dragging)
                return
            cameraYaw -= (mouse.x - lastX) * 0.45
            cameraPitch = Math.max(-89, Math.min(89, cameraPitch + (mouse.y - lastY) * 0.45))
            lastX = mouse.x
            lastY = mouse.y
        }
        onReleased: function(mouse) {
            if (mouse.button === Qt.LeftButton)
                dragging = false
        }
        onCanceled: dragging = false
        onWheel: function(wheel) {
            if (wheel.angleDelta.y === 0)
                return
            orthoHeight = Math.max(1.2, Math.min(80, orthoHeight * Math.pow(1.1, wheel.angleDelta.y / 120)))
            wheel.accepted = true
        }
    }

    Component.onCompleted: {
        updateCamera()
        emitCameraChanged()
    }
}
