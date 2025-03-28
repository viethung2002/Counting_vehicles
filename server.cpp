#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>

#pragma comment(lib, "ws2_32.lib") // Link thư viện WinSock

#define PORT 8081
#define BUFFER_SIZE 1024

int main() {
    WSADATA wsaData;
    SOCKET serverSocket, clientSocket;
    sockaddr_in serverAddr, clientAddr;
    char buffer[BUFFER_SIZE];

    // 1️⃣ Khởi tạo Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed! Error: " << WSAGetLastError() << "\n";
        return 1;
    }

    // 2️⃣ Tạo socket cho server
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed! Error: " << WSAGetLastError() << "\n";
        WSACleanup();
        return 1;
    }

    // 3️⃣ Cấu hình địa chỉ server
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY; // Chấp nhận tất cả IP
    serverAddr.sin_port = htons(PORT);

    // 4️⃣ Gán socket với địa chỉ và cổng
    if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed! Error: " << WSAGetLastError() << "\n";
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }

    // 5️⃣ Lắng nghe kết nối
    if (listen(serverSocket, 5) == SOCKET_ERROR) {
        std::cerr << "Listen failed! Error: " << WSAGetLastError() << "\n";
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }
    std::cout << "Server is listening on port " << PORT << "...\n";

    // 6️⃣ Chấp nhận kết nối từ client
    int clientAddrSize = sizeof(clientAddr);
    std::cout << "Waiting for client connection...\n";
    clientSocket = accept(serverSocket, (sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket == INVALID_SOCKET) {
        std::cerr << "Accept failed! Error: " << WSAGetLastError() << "\n";
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }
    std::cout << "Client connected!\n";

    // 7️⃣ Nhận dữ liệu từ client
    int recvBytes = recv(clientSocket, buffer, BUFFER_SIZE, 0);
    if (recvBytes > 0) {
        buffer[recvBytes] = '\0';
        std::cout << "Client gửi: " << buffer << std::endl;
    } else if (recvBytes == 0) {
        std::cout << "Client đã đóng kết nối.\n";
    } else {
        std::cerr << "recv() failed! Error: " << WSAGetLastError() << "\n";
    }

    // 8️⃣ Gửi phản hồi về client
    const char* response = "Hello from server!";
    send(clientSocket, response, strlen(response), 0);
    std::cout << "Response sent to client.\n";

    // 9️⃣ Đóng kết nối
    closesocket(clientSocket);
    closesocket(serverSocket);
    WSACleanup();
    std::cout << "Server shut down.\n";

    return 0;
}
