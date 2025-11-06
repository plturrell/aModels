package discoverability

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"
)

// NotificationService provides real-time notifications for marketplace events.
type NotificationService struct {
	db     *sql.DB
	logger *log.Logger
}

// NewNotificationService creates a new notification service.
func NewNotificationService(db *sql.DB, logger *log.Logger) *NotificationService {
	return &NotificationService{
		db:     db,
		logger: logger,
	}
}

// NotificationType represents the type of notification.
type NotificationType string

const (
	NotificationTypeAccessRequest    NotificationType = "access_request"
	NotificationTypeAccessGranted   NotificationType = "access_granted"
	NotificationTypeAccessDenied    NotificationType = "access_denied"
	NotificationTypeProductUpdated   NotificationType = "product_updated"
	NotificationTypeNewProduct      NotificationType = "new_product"
	NotificationTypeTagAdded        NotificationType = "tag_added"
)

// Notification represents a notification.
type Notification struct {
	ID        string          `json:"id"`
	UserID    string          `json:"user_id"`
	Type      NotificationType `json:"type"`
	Title     string          `json:"title"`
	Message   string          `json:"message"`
	ProductID string          `json:"product_id,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Read      bool            `json:"read"`
	CreatedAt time.Time       `json:"created_at"`
}

// CreateNotification creates a new notification.
func (ns *NotificationService) CreateNotification(ctx context.Context, notification *Notification) error {
	if notification.ID == "" {
		notification.ID = generateID()
	}
	if notification.CreatedAt.IsZero() {
		notification.CreatedAt = time.Now()
	}

	// In production, would store in database and send via WebSocket/SSE
	if ns.logger != nil {
		ns.logger.Printf("Notification created: %s for user %s", notification.Type, notification.UserID)
	}

	return nil
}

// NotifyAccessRequest creates a notification for an access request.
func (ns *NotificationService) NotifyAccessRequest(ctx context.Context, productID, requesterID, ownerID string) error {
	notification := &Notification{
		UserID:    ownerID,
		Type:      NotificationTypeAccessRequest,
		Title:     "New Access Request",
		Message:   fmt.Sprintf("User %s requested access to product %s", requesterID, productID),
		ProductID: productID,
		Data: map[string]interface{}{
			"requester_id": requesterID,
		},
	}

	return ns.CreateNotification(ctx, notification)
}

// NotifyAccessGranted creates a notification for granted access.
func (ns *NotificationService) NotifyAccessGranted(ctx context.Context, productID, requesterID string) error {
	notification := &Notification{
		UserID:    requesterID,
		Type:      NotificationTypeAccessGranted,
		Title:     "Access Granted",
		Message:   fmt.Sprintf("Your access request for product %s has been granted", productID),
		ProductID: productID,
	}

	return ns.CreateNotification(ctx, notification)
}

// NotifyNewProduct creates a notification for a new product.
func (ns *NotificationService) NotifyNewProduct(ctx context.Context, productID, productName, team string, subscribers []string) error {
	for _, userID := range subscribers {
		notification := &Notification{
			UserID:    userID,
			Type:      NotificationTypeNewProduct,
			Title:     "New Product Available",
			Message:   fmt.Sprintf("New product %s is now available in the marketplace", productName),
			ProductID: productID,
			Data: map[string]interface{}{
				"team": team,
			},
		}

		if err := ns.CreateNotification(ctx, notification); err != nil {
			if ns.logger != nil {
				ns.logger.Printf("Failed to notify user %s: %v", userID, err)
			}
		}
	}

	return nil
}

// GetNotifications retrieves notifications for a user.
func (ns *NotificationService) GetNotifications(ctx context.Context, userID string, limit, offset int) ([]*Notification, error) {
	// In production, would query database
	// For now, return empty list
	return []*Notification{}, nil
}

// MarkAsRead marks a notification as read.
func (ns *NotificationService) MarkAsRead(ctx context.Context, notificationID, userID string) error {
	// In production, would update database
	return nil
}

func generateID() string {
	return fmt.Sprintf("notif-%d", time.Now().UnixNano())
}

